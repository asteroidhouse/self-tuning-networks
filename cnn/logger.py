import os
import sys
import csv
import ipdb
import time
from datetime import datetime


class Logger():
    
    def __init__(self, argv, args, label_dict):
        
        self.save = args.save
        if not args.save:
            return

        # Construct log directory name and create log directory if it doesn't exist.
        save_str = self.create_save_str(argv)
        if args.subdir == '':
            date = datetime.utcnow().strftime("%Y%m%d")
            args.subdir = date
        time = datetime.utcnow().strftime("%H:%M:%S")
        self.logdir = os.path.join(args.logdir, args.dir, args.subdir, '{}_{}'.format(save_str, time))
        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)
        
        # Write experiment parameters/current git commit hash into a file in log directory.
        args_dict = vars(args)
        args_list = sorted(list(args_dict.keys()))
        with open(self.logdir + '/experiment_params.txt', 'w') as file:
            file.write('Logdir: ' + str(self.logdir) + '\n')
            for arg in args_list:
                file.write(arg + ': ' + str(args_dict[arg]) + '\n')

        # Write labels at the start of the csv file.
        self.csv_files_and_writers = {}
        for name in label_dict:
            filename = self.logdir + '/' + name + '.csv'
            csv_file = open(filename, 'w')
            writer = csv.DictWriter(csv_file, fieldnames=label_dict[name])
            
            self.csv_files_and_writers[name] = (csv_file, writer)
            writer.writeheader()
            csv_file.flush()

    def create_save_str(self, argv):
        argvals_to_write = []
        for i, arg in [(i, arg) for (i, arg) in enumerate(argv) if arg.count('-') == 1]:
            argval = arg.lstrip('-')
            if i+1 < len(argv) and argv[i+1].count('-') == 0:
                argval += argv[i+1]
            argvals_to_write.append(argval)
        
        argvals_to_write = sorted(argvals_to_write)
        save_str = '_'.join(argvals_to_write)
        if save_str == '':
            save_str = 'no_args'
        return save_str

    def write(self, name, stats):
        """
        Write given stats into the csv file.
        """
        if self.save:
            file, writer = self.csv_files_and_writers[name]
            writer.writerow(stats)
            file.flush()

    def close(self):
        for name in self.csv_files_and_writers:
            self.csv_files_and_writers[name][0].close()

