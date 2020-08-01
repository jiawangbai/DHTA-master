import warnings
import argparse
import torch.optim
import torch.utils.data
from model import *
from dataset import *
from evaluation import gen_params, mean_average_precision
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser(description='Target Attack for Deep Hashing based Retrieval')

parser.add_argument('--arch', dest='arch', default='vgg11',
                    help='hashing model architecture: (default: vgg11)')
parser.add_argument('--dataset', '--da', dest="dataset",
                    default="imagenet", choices=["imagenet"], type=str)
parser.add_argument('--root', dest='root', type=str, default='/data/dataset',
                    help='root path of your data.')
parser.add_argument('--path', dest='path', type=str, default='models/imagenet_vgg11_32',
                    help='path of model and hash codes.')

parser.add_argument('--n-bits', dest='n_bits', type=int, default=32)
parser.add_argument('--n-anchor', dest='n_an', type=int, default=9,
                    help='size of object set to generate the anchodr code.')
parser.add_argument('--epsilon', dest='eps', type=float, default=0.032)

parser.add_argument('--reproduce', '--r', dest='reproduce', action='store_true', default=False,
                    help='reproduce our results reported in the paper.')

parser.add_argument('--gpu-id', dest='gpu_id', type=str, default='0')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id


# loss function of DHTA
def target_adv_loss(clean_output, noisy_output, target_hash):

    k = clean_output.shape[-1]
    loss = -(noisy_output).mm(target_hash.t()).sum() / (k*len(target_hash))

    return loss


# targeted attack for single query
def target_adv(model, query, target_hash,
        epsilon=0.032, alpha=1, num_iter=2000):

    delta = torch.zeros_like(query, requires_grad=True)

    clean_output = model(query)

    for i in range(num_iter):

        factor = get_factor(i)
        loss = target_adv_loss(clean_output, model.forward_factor(query+delta, factor), target_hash)
        loss.backward(retain_graph=True)

        delta.data = delta - alpha * delta.grad.detach() / torch.norm(delta.grad.detach(), 2)

        delta.data = clamp(delta, query).clamp(-epsilon, epsilon)
        delta.grad.zero_()

    return delta.detach()


# according to the number of iterations, return the alpha
def get_factor(n):
    if n < 1000:
        return 0.1
    elif 1000 <= n < 1200:
        return 0.2
    elif 1200 <= n < 1400:
        return 0.3
    elif 1400 <= n < 1600:
        return 0.5
    elif 1600 <= n < 1800:
        return 0.7
    elif 1800 <= n < 2000:
        return 1


# clamp delta to image space
def clamp(delta, clean_imgs):
    MEAN = torch.tensor([[[0.485]], [[0.456]], [[0.406]]]).cuda()
    STD = torch.tensor([[[0.229]], [[0.224]], [[0.225]]]).cuda()

    clamp_imgs = (((delta.data + clean_imgs.data) * STD + MEAN) * 255).clamp(0, 255)
    clamp_delta = (clamp_imgs/255 - MEAN) / STD - clean_imgs.data

    return clamp_delta


# calculate anchor code using Component-voting Scheme
def get_anchor_code(codes):
    return np.sign(np.sum(codes, axis=0))


# evaluate DHTA on image hashing
def eval_target_adv():

    arch = args.arch
    dataset = args.dataset
    n_bits = args.n_bits
    n_anchor = args.n_an
    epsilon = args.eps
    path = args.path

    model = VGGHash(arch, n_bits).cuda()
    checkpoint = torch.load(os.path.join(path, "model.th"))
    model.load_state_dict(checkpoint['state_dict'])

    query_loader = torch.utils.data.DataLoader(
        dataset=Dataset(root=args.root, data_name=dataset, reproduce=args.reproduce),
        batch_size=1, shuffle=False, pin_memory=True)

    file = np.loadtxt(os.path.join(path, "database_hash.txt"))
    database_labels, database_hash = file[:, :100], file[:, 100:]

    query_labels = []
    for _, lab in query_loader:
        query_labels.append(lab[0].numpy().tolist())
    query_labels = np.array(query_labels)

    # if true, load target labels and samples indexes
    # to generate anchor codes to reproduce our result.
    # else, choose randomly.
    if args.reproduce:
        target_query_labels = np.loadtxt("reproduce/target_labels.txt")
        anchor_generation_indexes = np.loadtxt("reproduce/anchor_generation_indexes.txt").astype(int)

        target_hash = []
        for choose_indexes in anchor_generation_indexes:
            target_hash.append(get_anchor_code(np.array(database_hash)[choose_indexes])[np.newaxis, :].tolist())
        target_hash = np.array(target_hash).astype(np.float32)
    else:
        # choose the target label from labels which are different from original labels.
        db_str = []
        for l in database_labels:
            db_str.append("")
            for b in l:
                db_str[-1] += str(int(b))
        str2count = {}
        str2lab = {}
        strlist = []
        for s in db_str:
            if str2count.get(s) == None:
                str2count[s] = 1
            else:
                str2count[s] += 1
            if str2lab.get(s) == None:
                str2lab[s] = [int(i) for i in s]
                strlist.append(s)

        target_query_labels = []
        for i in range(len(query_labels)):
            l_str = ""
            for b in query_labels[i]:
                l_str += str(int(b))
            can_labels = [k for k in strlist if k != l_str and str2count[k] > args.n_an]
            tmp_str = np.random.choice(can_labels)
            target_query_labels.append(str2lab[tmp_str])
        target_query_labels = np.array(target_query_labels)

        # choose samples to generate the anchor code.
        target_hash = []
        for i in range(len(query_labels)):
            candidates = np.array([index for index, l in enumerate(database_labels)
                                   if (target_query_labels[i]==l).all()])
            choose_indexes = candidates[np.random.choice(len(candidates), n_anchor, replace=False)]
            target_hash.append(get_anchor_code(np.array(database_hash)[choose_indexes])[np.newaxis, :].tolist())
        target_hash = np.array(target_hash).astype(np.float32)

    attack_query_hash = []
    clean_query_hash = []
    model.eval()
    for i, (input, target) in enumerate(query_loader):
        if i % 10 == 0:
            print("Attack Sample: {0}".format(i))

        input_var = torch.autograd.Variable(input).cuda()

        # target_label = np.argwhere(target_query_labels == 1)[0][1]
        # la = np.argwhere(target_query_labels[i]==1)[0][0]
        th = torch.tensor(target_hash[i]).cuda()
        delta = target_adv(model, input_var, th, epsilon=epsilon)
        attack_hash = torch.sign(model(input_var+delta))
        attack_query_hash.append(attack_hash[0].cpu().detach().numpy().tolist())
        clean_query_hash.append(torch.sign(model(input_var)[0]).detach().cpu().numpy().tolist())
        # attack_query_hash.append(target_hash[i][0].tolist())

    attack_query_hash = np.array(attack_query_hash)
    clean_query_hash = np.array(clean_query_hash)

    R = 1000

    original_map = mean_average_precision(gen_params(database_hash, database_labels, clean_query_hash, query_labels, R=R))
    orignal_t_map = mean_average_precision(gen_params(database_hash, database_labels, clean_query_hash, target_query_labels, R=R))

    attack_map = mean_average_precision(gen_params(database_hash, database_labels, attack_query_hash, query_labels, R=R))
    attack_t_map = mean_average_precision(gen_params(database_hash, database_labels, attack_query_hash, target_query_labels, R=R))

    print("\nDataset: {0}  #Bits: {1}".format(dataset, n_bits))
    print("Origianl MAP: {0:.2f}, Origianl t-MAP: {1:.2f}, Attack MAP: {2:.2f}, Attack t-MAP: {3:.2f}".format(
        original_map*100, orignal_t_map*100, attack_map*100, attack_t_map*100))


if __name__ == "__main__":
    eval_target_adv()
