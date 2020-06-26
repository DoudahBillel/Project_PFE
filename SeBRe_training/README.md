# Project based on the Keras Project Template [![CometML](https://img.shields.io/badge/comet.ml-track-brightgreen.svg)](https://www.comet.ml)

Re-ordered SeBRe to more structured and fits the paradigms defined by the template. 
I added several thing, for example the usage of `Click` for parameter parsing. 
I removed the trainer class as the data is huge and can just occupy more memory if copied in another class instance.

# Getting Started

## Installing dependencies

All the dependencies are listed in the file `requirements.txt`. You can install it using pip.

```shell
pip install -r requirements.txt
```

Once every thing is install you can run the `main.py` to check if can be run
```shell
$python main.py --help
```

If everything is ok, you see this message:

```shell
Usage: main.py [OPTIONS]

Options:
  -c, --config PATH    Path to json configuration file  [required]
  -s, --split          Splits the raw images into training and validation sets
  -p, --percent FLOAT  The percentage (between 0 and 1) of data that will be
                       used for training. The rest is used for validation.
                       Default: 0.75
  --help               Show this message and exit.
```

## Project configuration

All the configuration parameters can be set by mofying (or creating a copy) of the file `configs/SeBRe_config.json`. These paramters will loaded by the `utils/config.py` which will return a `Config` class. The base `Config` class is defined in `base/base_config.py`.

## Splitting data

You need to edit the `configs/SeBRe_config.json` to tell it where the raw data is and where you want to generate the training anf validation folder.

By default they are configured to be in the `data` folder as follows:

```shell
data
├── Train
│   ├── Training_JPG
│   └── Training_Mask
├── Transformed
│   ├── TransformedJPG
│   └── TransformedMasks
└── Val
    ├── Val_JPG
    └── Val_Mask
```

You can then generated the data as follow:

```r
python -m pdb main.py -c configs/SeBRe_config.json -s -p 0.8
```

*NB:* you can pass any configuration file as long as it has the same structure as `SeBRe_config.json`

# Training

Just simply run the following command once all the data is ready:

```shell
python main.py -c configs/SeBRe_config.json
```

