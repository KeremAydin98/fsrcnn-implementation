from data import T91Dataset
from model import create_model, get_callbacks
import config

def training():


    train_data = T91Dataset(batch_size=3,
                             which_type="train",
                            color_channels=config.COLOR_CHANNELS)


    val_data = T91Dataset(batch_size=3,
                           which_type="validation",
                          color_channels=config.COLOR_CHANNELS)


    model = create_model(d=config.d,
                         s=config.s,
                         m=config.m,
                         rescaling=config.RESCALING_FACTOR,
                         color_channels=config.COLOR_CHANNELS)

    history = model.fit(train_data, validation_data=val_data, callbacks=get_callbacks(), epochs=200)


if __name__ == "__main__":

    training()




