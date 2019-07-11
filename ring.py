from header import *
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
world_size = comm.Get_size()
torch.manual_seed(rank * 7 + 13)
mode = MPI.MODE_WRONLY|MPI.MODE_CREATE|MPI.MODE_APPEND
from parser import *
from datasets.cifar_100 import *



from log import *
args.model = "ring"
extra_flag = True
use_cuda = args.cuda
logger = Logger(Comm=comm,Mode=mode,Middle_name=middle_name)
start_epoch = 0
# Model
if args.data_distribution == "iid":
    trainset,trainloader = generating_dataset_train_iid(rank = rank, world_size= world_size,model=args.model, training_batch_size = args.training_batch_size)
else:
    trainset,trainloader = generating_dataset_train_noniid(rank = rank, world_size= world_size,model=args.model, training_batch_size = args.training_batch_size)

testset,testloader = generating_dataset_test(testing_batch_size = args.testing_batch_size)
if args.resume:
    # Load checkpoint.
    
    # load specific checkpoint file
    middle_name = ""
    
    
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt {} .t7'.format(middle_name))
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    start_time = checkpoint['start_time']
else:
    logger.log_message('>>> Building model..')
    # net = vgg11()
    # net = torchvision.models.AlexNet(num_classes=10)
    # net = alexnet()
    net = LeNet() # you do not want to use any batchnorm in your network for extrasgd
    # net = VGG('VGG11')
    # net = ResNet18()
    # net = PreActResNet18()
    # net = GoogLeNet()
    # net = DenseNet121()
    # net = ResNeXt29_2x64d()
    # net = MobileNet()
    # net = MobileNetV2()
    # net = DPN92()
    # net = ShuffleNetG2()
    # net = SENet18()
    
    
if use_cuda:
    #seperate the work load to all the gpus
    cuda_rank = rank % torch.cuda.device_count()
    net.cuda(cuda_rank)
    # open this can seperate to all gpu
    #net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

    
def flatten_params():
    global net
    result = torch.cat([param.data.view(-1) for param in net.parameters()], 0).cpu().numpy()
    return result

def load_params(flattened):
    # if args.debug:
    #     log_message(">>> loading parameters... {}".format(flattened[-10:]))
    global net
    offset = 0
    for param in net.parameters():
        param.data.copy_(torch.from_numpy(flattened[offset:offset + param.nelement()]).view(param.size()))
        offset += param.nelement()

class DSGD(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate

    """

    def __init__(self, params, lr, extra=False):
        defaults = dict(lr=lr, extra=extra)
        super(DSGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(DSGD, self).__setstate__(state)

    def step(self):
        """Performs a single optimization step.
        """
        loss = None
        global current_epoch
        for group in self.param_groups:
            extra = group['extra']
            if args.decrease_lr == True:
                lr = group['lr'] * (0.99)**(current_epoch)
            else:
                lr = group['lr']
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if extra:
                    param_state = self.state[p]
                    if 'extra_gradient_buffer' not in param_state:
                        buf_grad = param_state['extra_gradient_buffer'] = torch.zeros_like(p.data)
                        buf_grad.add_(d_p)
                        buf_weights = param_state['extra_weights_buffer'] = torch.zeros_like(p.data)
                        buf_weights.add_(p.data)
                        #p.data.add_(-group['lr'], d_p)
                        p.data.add_(-lr,d_p)
                    else:
                        buf_grad = param_state['extra_gradient_buffer']
                        buf_weights = param_state['extra_weights_buffer']
                        saved_pdata = torch.zeros_like(p.data)
                        saved_pdata.copy_(p.data)
                        #p.data.add_(-group['lr'] * (d_p - buf_grad) - buf_weights + p.data)
                        p.data.add_(-lr*(d_p - buf_grad) - buf_weights + p.data)
                        buf_weights.copy_(saved_pdata)
                        buf_grad.copy_(d_p)
                else:
                    #p.data.add_(-lr,d_p)
                    p.data.add_(-group['lr'], d_p)

        after_update_hook()
        # for group in self.param_groups:
        #     extra = group['extra']
        #     if extra:
        #         for p in group['params']:
        #             param_state = self.state[p]
        #             buf_weights = param_state['extra_weights_buffer']
        #             buf_weights.copy_(p.data)

        return loss

    
    
    
    
def optionsgd(neighbors_position = [-1,1],weights=[1/3,1/3]):
    global current_epoch
    global iterations
    if args.dropout == True:
        dropout_list = dropout_list_list[iterations]
        iterations += 1
    if args.debug:
        logger.log_message(">>> dpsgd averaging...")
    if args.dropout == True:
        logger.log_message('dropouting------------')
        if len(dropout_list) < len(neighbors_position)+1:
            return 
        if rank in dropout_list:
            flatten_w = flatten_params()
            neighbors_w = [np.zeros_like(flatten_w) for x in range(len(neighbors_position))]
            neighbors_rank = [dropout_list[(dropout_list.index(rank) + position)%len(dropout_list)] for position in neighbors_position]
            send_reqs = [comm.Isend(flatten_w,dest=neighbors_rank[index]) for index in range(len(neighbors_rank))]
            [comm.Recv(neighbors_w[index],source = neighbors_rank[index]) for index in range(len(neighbors_rank))]
            [send_reqs[index].wait() for index in range(len(neighbors_rank))]
            composite_weighted_neighbors_w = [neighbors_w[index]*weights[index] for index in range(len(neighbors_rank))]
            average_w = reduce(lambda x,y:x+y,composite_weighted_neighbors_w) + flatten_w*(1-reduce(lambda x,y:x+y,weights))
            load_params(average_w)
    else:
        logger.log_message('not dropouting---')
        flatten_w = flatten_params()
        neighbors_w = [np.zeros_like(flatten_w) for x in range(len(neighbors_position))]
        neighbors_rank = [(rank+position)%world_size for position in neighbors_position]
        send_reqs = [comm.Isend(flatten_w,dest=neighbors_rank[index]) for index in range(len(neighbors_rank))]
        [comm.Recv(neighbors_w[index],source = neighbors_rank[index]) for index in range(len(neighbors_rank))]
        [send_reqs[index].wait() for index in range(len(neighbors_rank))]
        composite_weighted_neighbors_w = [neighbors_w[index]*weights[index] for index in range(len(neighbors_rank))]
        average_w = reduce(lambda x,y:x+y,composite_weighted_neighbors_w) + flatten_w*(1-reduce(lambda x,y:x+y,weights))
        load_params(average_w)
    
    
    
        
    
    
    
    
    
    
    
    
    
    
    
def after_update_hook():
    print(rank)
    global net
    if args.model == 'ps':
        global current_epoch
        flatten_w = flatten_params()
        print("central not dropout")
        if rank == 0:
            params_list = [np.zeros_like(flatten_w) for x in range(world_size)]
            print("start recv")
            [comm.Recv(params_list[x],source=x) for x in range(1,world_size)]
            print("recved rank0")
            sum_w = np.zeros_like(flatten_w)
            for m in params_list[1:]:
                sum_w += m
            average_w = (sum_w)/(world_size - 1)
            print("start_send rank0")
            send_reqs = [comm.Isend(average_w,dest = x) for x in range(1,world_size)]
            [req.wait() for req in send_reqs]
            print("finish sending rank0")
            load_params(average_w)
        else:
            params_w = np.zeros_like(flatten_w)
            print("req_sending--rank={}".format(rank))
            send_req = comm.Isend(flatten_w,dest = 0)
            send_req.wait()
            comm.Recv(params_w,source = 0)
            print("recv --rank={}".format(rank))
            load_params(params_w)
    else:
        raise NotImplementedError
        
        
        
criterion = nn.CrossEntropyLoss()
if args.model != 'ps' and extra_flag==True:
    optimizer = DSGD(net.parameters(), lr=args.lr, extra=True)
else:
    optimizer = DSGD(net.parameters(), lr=args.lr)

        
        
        
        
        
        
# Training
def train(epoch):
    global net
    logger.log_message('>>> Epoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if rank==0:
            after_update_hook()
            logger.log_message("rank0-{}".format(batch_idx))
        else:
            ## ?? what for
            global iterations
            if use_cuda:
                print("using cuda")
                inputs, targets = inputs.cuda(cuda_rank), targets.cuda(cuda_rank)
            optimizer.zero_grad()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()


            train_loss += loss.data.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            # averaging parameters
            print("update slave")
            #after_update_hook()
            logger.log_message('TRAIN -- epoch: {epoch} -- batch_index: {batch_idx} -- total: {total} -- loss: {loss} -- loss-avg: {loss_avg} -- accuracy: {acc}'
                        .format(epoch=epoch,
                                batch_idx = batch_idx,
                                total=len(trainloader),
                                loss = loss.data.item(),
                                loss_avg= train_loss/(batch_idx+1),
                                acc=100.*correct/total))


def test(epoch):
    # only rank 0 tests
    if rank != 0:
        return
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(cuda_rank), targets.cuda(cuda_rank)
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.data.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        logger.log_message('TEST -- epoch: {epoch} -- batch_index: {batch_idx} -- total: {total} -- loss: {loss} -- accuracy: {acc}'
                    .format(epoch=epoch,
                            batch_idx = batch_idx,
                            total=len(testloader),
                            loss= test_loss/(batch_idx+1),
                            acc=100.*correct/total))
    logger.log_res('epoch:{}  '.format(epoch)+'loss:{}  '.format(test_loss/args.testing_batch_size))
        # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #     % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    
    acc = 100.*correct/total
    if epoch%50==0:
        logger.log_message('Saving..')
        state = {
            'net': net,
            'acc': acc,
            'epoch': epoch,
            'start_time':start_time,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt {} .t7'.format(middle_name))
        best_acc = acc

        
import multiprocessing
current_epoch = 0
for epoch in range(start_epoch,500):
    #global current_epoch
    current_epoch = epoch
    train(epoch)
    comm.Barrier()
    if rank==0:
        multiprocessing.Process(target=test,args=(epoch,)).start()
    comm.Barrier()


fh.Sync()
fh.Close()

