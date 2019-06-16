import os
import csv
from collections import defaultdict

import matplotlib.pyplot as plt


def load_data(csv_filename):
    with open(csv_filename, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        fields = reader.fieldnames
        agg_dict = defaultdict(list)
        for row in reader:
            for key in fields:
                try:
                    value = int(row[key])
                except:
                    try:
                        value = float(row[key])
                    except:
                        value = row[key]
                agg_dict[key].append(value)
    return agg_dict


log_dir_init05 = 'logs/dropouto_schedule_init05/2019-06-15'
val_dict_init05 = load_data(os.path.join(log_dir_init05, 'val.csv'))
epoch_dict_init05 = load_data(os.path.join(log_dir_init05, 'epoch.csv'))

log_dir_init30 = 'logs/dropouto_schedule_init30/2019-06-15'
val_dict_init30 = load_data(os.path.join(log_dir_init30, 'val.csv'))
epoch_dict_init30 = load_data(os.path.join(log_dir_init30, 'epoch.csv'))

log_dir_init50 = 'logs/dropouto_schedule_init50/2019-06-15'
val_dict_init50 = load_data(os.path.join(log_dir_init50, 'val.csv'))
epoch_dict_init50 = load_data(os.path.join(log_dir_init50, 'epoch.csv'))

log_dir_init70 = 'logs/dropouto_schedule_init70/2019-06-15'
val_dict_init70 = load_data(os.path.join(log_dir_init70, 'val.csv'))
epoch_dict_init70 = load_data(os.path.join(log_dir_init70, 'epoch.csv'))

log_dir_init90 = 'logs/dropouto_schedule_init90/2019-06-15'
val_dict_init90 = load_data(os.path.join(log_dir_init90, 'val.csv'))
epoch_dict_init90 = load_data(os.path.join(log_dir_init90, 'epoch.csv'))

plt.figure(figsize=(8,6))
plt.plot(val_dict_init05['global_step'], val_dict_init05['dropouto'], label='Init=0.05', linewidth=2)
plt.plot(val_dict_init30['global_step'], val_dict_init30['dropouto'], label='Init=0.3', linewidth=2)
plt.plot(val_dict_init50['global_step'], val_dict_init50['dropouto'], label='Init=0.5', linewidth=2)
plt.plot(val_dict_init70['global_step'], val_dict_init70['dropouto'], label='Init=0.7', linewidth=2)
plt.plot(val_dict_init90['global_step'], val_dict_init90['dropouto'], label='Init=0.9', linewidth=2)
plt.ylim(0, 1)
plt.xlabel('Iteration', fontsize=26)
plt.ylabel('Output Dropout Rate', fontsize=26)
plt.xticks([0, 5000, 10000, 15000, 20000, 25000], ['0', '5k', '10k', '15k', '20k', '25k'], fontsize=20)
plt.yticks(fontsize=20)
plt.legend(fancybox=True, framealpha=0.5, fontsize=24)

if not os.path.exists('plots'):
    os.makedirs('plots')

plt.savefig('plots/stn_dropouto_schedules.pdf', bbox_inches='tight')
