
class config_train(object):


    batch_size = 64
    num_channels = 1
    num_classes = 10
    image_size = 32
    latent_dim = 128
    lr = 0.002
    weight_decay =1



class config_test(object):
    batch_size = 64
    num_channels = 1
    num_classes = 10
    image_size = 32
    latent_dim = 128




class directories(object):
    result_test="result_test"
    result_train="result_train"
    data = "dataset"