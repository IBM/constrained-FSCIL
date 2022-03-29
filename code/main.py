#
# Copyright 2022- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#

# ==================================================================================================
# IMPORTS
# ==================================================================================================

import argparse
import datetime
import os
import platform
from ast import literal_eval
from copy import deepcopy

import numpy as np
import pdb
if platform.system() == 'Darwin':
    # This had to be done to resolve a strange issue with OpenMP on MacOS
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# ==================================================================================================
# FUNCTIONS
# ==================================================================================================

# --------------------------------------------------------------------------------------------------
# Other
# --------------------------------------------------------------------------------------------------

def test_and_cast(key, value):
    integers = ['num_ways', 'num_ways_training', 'num_ways_inference', 'num_shots', 'num_shots_training',
                'num_shots_inference', 'num_query_training', 'num_query_inference','batch_size', 
                'batch_size_training', 'batch_size_inference','nudging_act_exp',
                'max_train_iter', 'max_val_iter', 'max_test_iter', 'validation_frequency', 'moving_average_samples',
                'summary_frequency_often', 'dim_features', 'summary_frequency_once', 'summary_frequency_very_often', 
                'summary_frequency_seldom', 'random_seed','retrain_iter','nudging_iter', 'lr_step_size', 'num_workers',
                'em_compression_nsup']
    floats = ['learning_rate', 'norm_weight', 'fixsup_weight', 'sharpening_strength', 'regularization', 'dropout_rate', 
               'SGDmomentum','SGDweight_decay']
    strings = ['block_architecture','block_interface','sharpening_activation', 'log_dir', 'representation', 'log',
               'data_folder', 'experiment_dir', 'loss_function','dataset','optimizer','trainstage', 'resume','pretrainFC',
               'retrain_act','nudging_act','em_compression']
    bools = ['with_repetition', 'allow_empty_classes', 'normalize_weightings','average_support_vector_inference', 'inference_only',
             'external_experiment', 'dropout', 'SGDnesterov','bipolarize_prototypes']
    integer_tuples = ['image_size', 'num_filters', 'kernel_sizes', 'maxpool_sizes', 'dense_sizes']
    float_tuples = ['dataset_split']

    if key in integers:
        value = int(value)
    elif key in floats:
        value = float(value)
    elif key in strings:
        pass
    elif key in bools:
        value = parse_bool(value)
    elif key in integer_tuples:
        value = literal_eval(value)
    elif key in float_tuples:
        value = literal_eval(value)
    else:
        raise KeyError("Parameter key case for \'{}\' not covered.".format(key))
    return key, value


def parse_bool(value):
    if value in ['True', 'true', 't', '1']:
        value = True
    elif value in ['False', 'false', 'f', '0']:
        value = False
    else:
        raise ValueError("Boolean value not recognized.")
    return value


def parse_answer(value):
    if value in ['Yes', 'yes', 'Y', 'y']:
        value = True
    elif value in ['No', 'no', 'N', 'n']:
        value = False
    else:
        raise ValueError("Boolean value not recognized.")

    return value


def create_log_str(args):
    # Log directory
    log_prefix = args.logprefix if args.logprefix \
        else '/dataP/man/log/test'
    log_suffix = args.logsuffix if args.logsuffix \
        else datetime.datetime.now().strftime('%Y_%m_%d/%H_%M_%S')
    log_dir = args.logdir if args.logdir \
        else log_prefix + '/' + log_suffix
    return log_dir


def round_to_pow2(x):
    """
    :param x:
    :type x: int
    :return:
    """
    return 1 << (x - 1).bit_length()


def is_overwritten(key, parameters, defaults):
    return True if parameters[key] != defaults[key] else False

# ==================================================================================================
# MAIN
# ==================================================================================================

# --------------------------------------------------------------------------------------------------
# Parser
# --------------------------------------------------------------------------------------------------

# Argument parser
verbose_parent_parser = argparse.ArgumentParser(add_help=False)
verbose_parent_parser.add_argument('-v', '--verbose', help='Increase output verbosity.', action='store_true')

submit_parent_parser = argparse.ArgumentParser(add_help=False)
submit_parent_parser.add_argument('--null', '-n', action='store_true',
                                  help='Use this option for dummy submission with output.')
submit_parent_parser.add_argument('--limited', '-l', action='store_true',
                                  help='Run a limited number of jobs in parallel.')
submit_parent_parser.add_argument('--queue', '-q', type=str, choices=['prod.short', 'prod.med', 'prod.long'],
                                  help='Specify the queue to submit the jobs to.')
submit_parent_parser.set_defaults(queue='prod.med')

log_parent_parser = argparse.ArgumentParser(add_help=False)
log_group = log_parent_parser.add_mutually_exclusive_group()
log_group.add_argument('-lp', '--logprefix', type=str, help='Specify a logging path prefix (i.e. root).')
log_group.add_argument('-ls', '--logsuffix', type=str, help='Specify a logging subdirectory.')
log_group.add_argument('-ld', '--logdir', type=str, help='Specify the whole logging path.')

main_parser = argparse.ArgumentParser(parents=[verbose_parent_parser])
# subparsers = main_parser.add_subparsers(help='Test', required=True)
subparsers = main_parser.add_subparsers()

# Experiment parser
experiment_parser = subparsers.add_parser('experiment', parents=[],
                                          help='Finish PCM experiments that have been run on the experimental platform.'
                                          )

experiment_parser.add_argument('logdir', type=str,
                               help='Specify the logging directory where the memory outputs are stored.'
                               )
experiment_parser.add_argument('--verify', '-y', action='store_true',
                               help='Simulation with lossless external memory.'
                               )
experiment_parser.set_defaults(which='experiment')

# Simulation parser
simulation_parser = subparsers.add_parser('simulation', parents=[log_parent_parser, verbose_parent_parser],
                                          help='Start a simulation of the model.',
                                          )
simulation_parser.add_argument('-p', '--parameter', action='append', nargs=2, metavar=('key', 'value'),
                               help='Specify parameters with key-value pairs. Chainable.'
                               )
simulation_parser.set_defaults(which='simulation')


# Cleanup parser
cleanup_parser = subparsers.add_parser('cleanup',
                                       parents=[verbose_parent_parser],
                                       help='Clean up a logging directory structure.',
                                       )
cleanup_parser.add_argument('function', type=str, choices=['unfinished', 'checkpoints'],
                            help='Choose what to clean up. "unfinished" deletes unfinished runs, "checkpoint" deletes trainable parameter backup files.'
                            )
cleanup_parser.add_argument('logdir', type=str, help='Specify the root to clean up.')
cleanup_parser.set_defaults(which='cleanup')

# Evaluation parser
evaluation_parser = subparsers.add_parser('evaluation',
                                          parents=[verbose_parent_parser],
                                          help='Use after simulations have been run.',
                                          )
evaluation_parser.add_argument('function', type=str, choices=['gather', 'test', 'copy_best'],
                               help='Use "gather" to collect all results in a single file.'
                               )
evaluation_parser.add_argument('logdir', type=str, help='Specify the root to collect the results from.')
evaluation_parser.add_argument('-b', '--best', action='store_true',
                               help=''
                               )
evaluation_parser.set_defaults(which='evaluation')

# Submit parser
submit_parser = subparsers.add_parser('submit',
                                      parents=[verbose_parent_parser],
                                      help='Use to dispatch multiple simulations specified in "submitter.py".',
                                      )
# submit_subparsers = submit_parser.add_subparsers(help='Test', required=True)
submit_subparsers = submit_parser.add_subparsers()

# Simulation submitter
submit_simulation_parser = \
    submit_subparsers.add_parser('simulations',
                                 parents=[submit_parent_parser, verbose_parent_parser],
                                 help='Submit multiple simulations to the cluster.',
                                 )
submit_simulation_parser.add_argument('--indices', '-i', type=int, nargs='+',
                                      help='Specify indices of "parameter_list" that should be submitted. Use spaces to separate indices.'
                                      )
submit_simulation_parser.add_argument('--shuffle', '-s', action='store_true',
                                      help='In case the submission order should be shuffled.'
                                      )
submit_simulation_parser.set_defaults(which='submit_simulations')

# Experiment submitter
submit_simulation_parser = \
    submit_subparsers.add_parser('experiments',
                                 help='Finish multiple experiments.',
                                 parents=[submit_parent_parser, verbose_parent_parser])
submit_simulation_parser.add_argument('root', type=str,
                                      help='Specify the root of the experiments.',
                                      )
submit_simulation_parser.add_argument('--verify', '-y', action='store_true',
                                      help='Simulation with lossless external memory.'
                                      )
submit_simulation_parser.set_defaults(which='submit_experiments')

# Parse arguments
args = main_parser.parse_args()

# --------------------------------------------------------------------------------------------------
# Execution
# --------------------------------------------------------------------------------------------------

# TODO: Make packages and provide them as arguments in the parser
# TODO: Solve dependent parameters nicer

if args.which in ['simulation', 'experiment']:
    # Define default parameters that are allowed to be called
    parameters = {
        # Architecture parameters
        ## Network
        'block_architecture':               'mini_resnet12',  # 'stanford'
        'block_interface':                  'GAP_FC',
        'resblocknorm':                     "none",
        'num_filters':                      None,
        'kernel_sizes':                     None,
        'maxpool_sizes':                    None,
        'dense_sizes':                      (1024,),
        ## Model
        'dim_features':                     512,
        'sharpening_activation':            'softabs',  # 'softabs', 'softrelu', 'abs', 'relu', 'exp'
        'sharpening_strength':              10.,
        'representation':                   'real',
       
        # Hardware approximations
        'normalize_weightings':             True,

        # Retraining operations
        'retrain_iter':                     20,
        'nudging_iter':                     20,
        'nudging_act':                      'doubleexp',
        'nudging_act_exp':                  4,
        'retrain_act':                      'tanh',
        'bipolarize_prototypes':            False,
        
        # Dataset parameters
        'dataset':                          'mini_imagenet',
        'data_folder':                      './data/miniimagenet',
        'image_size':                        (3,84, 84),
        'with_repetition':                  True,
        'allow_empty_classes':              True,
        'dataset_split':                    (0.85, 0.15),
        'random_seed':                      None,
        'num_workers':                      4,

        # Optimization parameters
        'trainstage':                       "pretrain_baseFSCIL",
        'optimizer':                        "Adam", # ???
        'SGDmomentum':                      0.9,
        'SGDweight_decay':                  5e-4, 
        'SGDnesterov':                      True,
        'learning_rate':                    1e-4,
        'lr_step_size':                     30000,
        'norm_weight':                      10,
        'fixsup_weight':                    0.1, # ???
        'dropout':                          False,
        'dropout_rate':                     0.0,
        'dropout_rate_interm':              0.0,
        'loss_function':                    'log',  # 
        'pretrainFC':                       'linear',

        # Problem parameters
        'num_ways':                         5,
        'num_shots':                        1,
        'num_ways_training':                None,
        'num_shots_training':               None,
        'num_query_training':               None,
        'num_ways_inference':               None,
        'num_shots_inference':              None,
        'num_query_inference':              None,

        # Test/training parameters
        'batch_size':                       None,
        'batch_size_training':              None,
        'batch_size_inference':             None,
        'max_train_iter':                   30000,
        'max_val_iter':                     20, 
        'max_test_iter':                    1000,
        'validation_frequency':             500,

        # Compression parameters
        'em_compression':                   'none', 
        'em_compression_nsup':              2,

        # Representation check parameters
        'check_representation_similarity':  False,
        'average_support_vector_inference': True,
        # Logging parameters
        'log_dir':                          None,
        'resume':                           '',
        'inference_only':                   False,
        'experiment_dir':                   None,
        'external_experiment':              False,
        'summary_frequency_seldom':         2500,
        'summary_frequency_often':          250,
        'summary_frequency_very_often':     10

    }

    # Dependent defaults
    parameters['batch_size'] = round_to_pow2(4 * parameters['num_ways'])

    if args.which == 'simulation':
        from lib.run_FSCIL import pretrain_baseFSCIL, train_FSCIL, metatrain_baseFSCIL
        #  Create the log directory path
        parameters['log_dir'] = create_log_str(args)

        # Store defaults to check what was overwritten
        defaults = deepcopy(parameters)

        # Check if parameter arguments are valid, cast the strings and update the parameters
        if args.parameter:
            for key, value in args.parameter:
                if key not in parameters:
                    raise KeyError('Not a valid parameter key: \"{}\".'.format(key))

                # Cast the strings and update the parameters
                key, value = test_and_cast(key, value)
                parameters.update({key: value})

        # --------------------------------------------------------------------------------------------------
        # Dependent Parameter Updates
        # --------------------------------------------------------------------------------------------------

        parameters['experiment_dir'] = parameters['log_dir'] + '/experiment'

        if parameters['num_filters'] is None:
            parameters['num_filters'] = (64,160,320,640)

        if parameters['batch_size_training'] is None:
            if parameters['batch_size']:
                parameters['batch_size_training'] = parameters['batch_size']
            else:
                raise ValueError
        if parameters['batch_size_inference'] is None:
            if parameters['batch_size']:
                parameters['batch_size_inference'] = parameters['batch_size']
            else:
                raise ValueError
        if parameters['num_ways_training'] is None:
            if parameters['num_ways']:
                parameters['num_ways_training'] = parameters['num_ways']
            else:
                raise ValueError
        if parameters['num_ways_inference'] is None:
            if parameters['num_ways']:
                parameters['num_ways_inference'] = parameters['num_ways']
            else:
                raise ValueError
        if parameters['num_shots_training'] is None:
            if parameters['num_shots']:
                parameters['num_shots_training'] = parameters['num_shots']
            else:
                raise ValueError
        if parameters['num_shots_inference'] is None:
            if parameters['num_shots']:
                parameters['num_shots_inference'] = parameters['num_shots']
            else:
                raise ValueError
        if parameters['num_query_training'] is None:
          if parameters['batch_size_training']:
            parameters['num_query_training'] = parameters['batch_size_training']
          else: 
            raise ValueError
        if parameters['num_query_inference'] is None:
          if parameters['batch_size_inference']:
            parameters['num_query_inference'] = parameters['batch_size_inference']
          else: 
            raise ValueError


        if parameters['trainstage']=="pretrain_baseFSCIL":
          func = pretrain_baseFSCIL
        elif parameters['trainstage']=="train_FSCIL":
          func = train_FSCIL
        elif parameters['trainstage']=="metatrain_baseFSCIL":
          func = metatrain_baseFSCIL

    # Remove unwanted parameters
    unwanted_parameters = ['batch_size', 'num_ways', 'num_shots']
    for unwanted_parameter in unwanted_parameters:
        parameters.pop(unwanted_parameter)

    # Set all unused parameters to None
    unused_parameters = []
    
    if not parameters['external_experiment']:
        unused_parameters += ['experiment_dir']

    if 'binary' not in parameters['representation']:
        unused_parameters += ['approximate_binary_similarity']

    if parameters['sharpening_activation'] not in ['softabs', 'softrelu','scaledexp']:
        unused_parameters += ['sharpening_strength']

    for unused_parameter in unused_parameters:
        parameters[unused_parameter] = None

    func(verbose=args.verbose, **parameters)

elif args.which == 'submit_simulations':
    from submitter import submit_simulations

    submit_simulations(indices=args.indices, shuffle=args.shuffle, queue=args.queue, limited=args.limited,
                       null=args.null)

elif args.which == 'submit_experiments':
    from submitter import submit_experiments, submit_verification

    if args.verify:
        submit_verification(args.root, queue=args.queue, limited=args.limited, null=args.null)
    else:
        submit_experiments(args.root, queue=args.queue, limited=args.limited, null=args.null)
else:
    raise ValueError