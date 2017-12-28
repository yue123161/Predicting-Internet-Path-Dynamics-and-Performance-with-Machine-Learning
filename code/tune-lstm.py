from utils import *
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-p', '--pkl_data_path', type=str)
parser.add_argument('-na', '--task_name', type=str)
parser.add_argument('-ts', '--time_steps', type=int, default=10)
parser.add_argument('-ep', '--epochs', type=int, default=300)
parser.add_argument('-bs', '--batch_size', type=int, default=8196)
parser.add_argument('-dr', '--dropout_rate', type=float, default=0.5)
parser.add_argument('-hd', '--hidden_dimension', type=int, default=128)

# save args
args = parser.parse_args()

LSTM_task(
        pkl_data_path=args.pkl_data_path,
        task_name=args.task_name+'+ep='+str(args.epochs)+'+bs='+str(args.batch_size)+'+dr='+str(args.dropout_rate)+'+hd='+str(args.hidden_dimension),
        time_steps=args.time_steps,
        epochs=args.epochs,
        batch_size=args.batch_size,
        dropout_rate=args.dropout_rate,
        hidden_dimension=args.hidden_dimension
    )
