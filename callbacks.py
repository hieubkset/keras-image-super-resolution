import os

from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint


class LRTensorBoard(TensorBoard):
    def __init__(self, log_dir, **kwargs):
        super().__init__(log_dir=log_dir, **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        logs.update({'lr': K.eval(self.model.optimizer.lr)})
        super().on_epoch_end(epoch, logs)


def make_tb_callback(log_dir):
    tb_callback = LRTensorBoard(log_dir=log_dir, write_graph=True)
    return tb_callback


def make_lr_callback(lr_init, lr_decay, lr_decay_at_steps):
    def lr_scheduler(epoch):
        lr = lr_init
        for decay_step in lr_decay_at_steps:
            lr = lr * (lr_decay ** ((epoch + 1) >= decay_step))
        return lr

    lr_callback = LearningRateScheduler(lr_scheduler)
    return lr_callback


class AltModelCheckpoint(ModelCheckpoint):
    def __init__(self, filepath, alternate_model, **kwargs):
        """
        Fix issue of saving a multiple gpu model
        https://github.com/keras-team/keras/issues/8123
        https://github.com/keras-team/keras/issues/8858
        """

        self.alternate_model = alternate_model
        super().__init__(filepath, **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        model_before = self.model
        self.model = self.alternate_model
        super().on_epoch_end(epoch, logs)
        self.model = model_before


def make_cp_callback(checkpoint_dir, model):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    checkpoint_path = os.path.join(checkpoint_dir, "cp-{epoch:04d}.h5")

    cp_callback = AltModelCheckpoint(checkpoint_path, model, save_weights_only=True, save_best_only=False, period=1)

    return cp_callback
