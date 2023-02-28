import torch
import torch.optim as optim

class My_Optimizer(object):

    def __init__(self, opt, model, lfm):
        """
        Initializes My_Optimizer object. 
        Params:
            opt: options 
            model: nn.model
            lfm: loss_function_manager object
        """
        self.lr_gamma = opt.lr_gamma
        self.max_norm = opt.max_norm

        # model optimizer optimizes the parameters of the future prediction nn
        if opt.optimizer == "adam":
            self.model_optimizer = optim.Adam(model.parameters(), 
                    lr=opt.lr, weight_decay=opt.weight_decay)
        elif opt.optimizer == "sgd":
            self.model_optimizer = optim.SGD(model.parameters(), 
                    lr=opt.lr, momentum=opt.momentum)
        self.lr_model = opt.lr


    def zero_grad(self):
        """
        Zeros gradients of an optimizer
        """
        self.model_optimizer.zero_grad()


    def grad_clip(self, model, lfm, fig_location, epoch, plot=False):
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.max_norm)


    def step(self):
        """
        Gradient step
        """
        self.model_optimizer.step()



    def backprop(self, loss_model):
        """
        """
        loss_model.backward()

    def state_dict(self):
        """
        """
        model_state = self.model_optimizer.state_dict()
        return (model_state)

    def lr_state(self):
        """
        """
        return self.lr_model

    def load_state_dict(self, state_tuple):
        """
        Load the state specified in state_tuple.
        Params:
           
        """
        self.model_optimizer.load_state_dict(state_tuple[0])

    def lr_decay(self):
        """
        Adjust learning rate according to the lr_gamma.
        """
        self.lr_model = self.lr_model * self.lr_gamma
        self.set_lr(self.lr_model)

    def load_lr(self, lr):
        self.lr_model = lr
        self.set_lr(self.lr_model)

        
    def set_lr(self, lr_model=None):
        if lr_model is None:
            lr_model = self.lr_model
        for param_group in self.model_optimizer.param_groups:
            param_group['lr'] = lr_model

