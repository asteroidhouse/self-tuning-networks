import os
import sys
import csv
import time
import datetime

# YAML setup
from ruamel.yaml import YAML
yaml = YAML()
yaml.preserve_quotes = True
yaml.boolean_representation = ['False', 'True']


class Logger():
    def __init__(self, argv, args, short_args={}, files=[], stats={}):

        self.save = args.save
        if not self.save: return

        exp_name = self.create_exp_name(args)
        self.log_dir = os.path.join('logs', args.save_dir, exp_name)

        # Check if the result file exists, and if so, don't run it again.
        if not args.overwrite:
            if os.path.exists(os.path.join(self.log_dir, 'result')) or os.path.exists(os.path.join(self.log_dir, 'result.csv')):
                print("The result file {} exists! Not rerunning.".format(os.path.join(self.log_dir, 'result')))
                sys.exit(0)

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # Save command-line arguments
        with open(os.path.join(self.log_dir, 'args.yaml'), 'w') as f:
            yaml.dump(vars(args), f)

        # Write code files into the log directory.
        for code_filename in files:
            code_file = os.path.join(os.path.dirname(__file__), code_filename)
            with open(code_file) as f:
                code = f.readlines()
            code_filename = code_filename.split("/")[-1]
            with open(os.path.join(self.log_dir, code_filename + ".copy"), "w") as file:
                for line in code:
                    file.write(line)

        # Write labels at the start of the csv file.
        self.csv_files_and_writers = {}
        for stat_type in stats:
            filename = os.path.join(self.log_dir, '{}.csv'.format(stat_type))
            csv_file = open(filename, 'w')
            writer = csv.DictWriter(csv_file, fieldnames=stats[stat_type])
            self.csv_files_and_writers[stat_type] = (csv_file, writer)
            writer.writeheader()
            csv_file.flush()

    def create_exp_name(self, args):
        timestamp = '{:%Y-%m-%d}'.format(datetime.datetime.now())

        if args.prefix:
            exp_name = '{}-{}'.format(timestamp, args.prefix)
        else:
            exp_name = timestamp

        return exp_name

    def write(self, stat_type, stats):
        """
        Write given stats into the csv file.
        """
        if self.save:
            file, writer = self.csv_files_and_writers[stat_type]
            writer.writerow(stats)
            file.flush()

    def close(self):
        for stat_type in self.csv_files_and_writers:
            self.csv_files_and_writers[stat_type][0].close()
