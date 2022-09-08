from data import T91_dataset
from model import create_model, get_callbacks
import config
def training():


    train_data = T91_dataset(batch_size=32,
                             type="train")

    val_data = T91_dataset(batch_size=32,
                           type="validation")

    test_data = T91_dataset(batch_size=32,
                            type="test")

    model = create_model(d=config.d,
                         s=config.s,
                         m=config.m,
                         rescaling=config.RESCALING_FACTOR)

    history = model.fit(train_data, validation_data=val_data, callbacks=get_callbacks(), epochs=200)


if __name__ == "main":

    training()




