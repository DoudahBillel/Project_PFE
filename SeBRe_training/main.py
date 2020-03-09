from data_loader.SeBRe_loader import SeBReDataLoader
from models.SeBRe_model import MaskRCNN
from utils.config import process_config
from utils.dirs import create_dirs
from utils.splitter import spliter

import click

def splitData(config, percent):
    spliter(config, percent).split_data()

def trainModel(config):

    # create the experiments dirs
    create_dirs([config.MODEL_DIR, config.MODEL_SAVE_DIR])

    click.echo(click.style('Loading training data ...',fg='green',bold=True))
    dataset_train = SeBReDataLoader(config = config, mode="training")
    dataset_train.load_brain()

    
    click.echo(click.style('Loading validation data ...',fg='green',bold=True))
    dataset_val = SeBReDataLoader(config= config, model ="validation")
    dataset_val.load_brain()
    
    click.echo(click.style('Preparing the model ...',fg='green',bold=True))
    model = MaskRCNN(mode="training", config=config, model_dir= config.MODEL_DIR)

    if not config.WEIGHTS_HDF5 is None:
        click.echo(click.style('Loading initial weights ...',fg='green',bold=True))
        model.load_weights(config.WEIGHTS_HDF5,by_name=True)

    click.echo(click.style('Training Head branches ...',fg='green',bold=True))
    # Passing layers="heads" freezes all layers except the head
    # layers. You can also pass a regular expression to select
    # which layers to train by name pattern.
    model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE, 
            epochs=1, 
            layers='heads') 
    
    click.echo(click.style('Fine tunning all layers ...',fg='green',bold=True))
    # Passing layers="all" trains all layers. You can also 
    # pass a regular expression to select which layers to
    # train by name pattern.
    model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE / 10,
            epochs=3, 
            layers="all")

    click.echo(click.style('Saving model weights ...',fg='green',bold=True))
    model.save()


@click.command()
@click.option('-c',"--config","config_file",required=True, type=click.Path(), help="Path to json configuration file")
@click.option('-s',"--split","split",is_flag=True, type=bool, help="Splits the raw images into training and validation sets")
@click.option('-p',"--percent","percent",required=False, default=0.75, type=float, help="The percentage (between 0 and 1) of data that will be used for training. The rest is used for validation. Default: 0.75")
def main(config_file, split, percent):
    # capture the config path from the run arguments
    # then process the json configuration file
    try:        
        config = process_config(config_file)
        config.display()
    except:
        raise Exception("Error while processing the config file")

    if split:
        splitData(config, percent)
    else:
        trainModel(config)

if __name__ == '__main__':
    main()
