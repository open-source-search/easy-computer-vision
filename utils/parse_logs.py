import sys
import os
import subprocess
import argparse


class ParseLogs():

    TEE = "tee"
    LOGS_DIR = "logs/"
    EPOCH = "epoch"
    VALUE = "value"
    LOSS = "loss"
    VAL_LOSS = "val_loss"

    @staticmethod
    def redirect_logs_to_file(logs_path):
        tee = subprocess.Popen([ParseLogs.TEE, logs_path], stdin=subprocess.PIPE)
        os.dup2(tee.stdin.fileno(), sys.stdout.fileno())
        os.dup2(tee.stdin.fileno(), sys.stderr.fileno())

    def calculate_best_model(self, file_path=None, dir_path=None):
        if dir_path:
            self.LOGS_DIR = dir_path
        if not file_path:
            for (dirpath, dirnames, filenames) in os.walk(self.LOGS_DIR):
                log_files = filenames
                break
        else:
            self.LOGS_DIR = os.path.dirname(file_path)
            log_files = [os.path.basename(file_path)]
        for log_file in log_files:
            splits = []
            output_dict = {}
            with open(os.path.join(self.LOGS_DIR, log_file), "r") as file:
                for line in file.readlines():
                    if line.lower().__contains__(self.EPOCH+" "):
                        epoch = line.strip()
                    if line.lower().__contains__(self.VAL_LOSS):
                        splits.append({
                                        self.EPOCH: epoch,
                                        self.LOSS: {element.split(":")[0].strip():
                                                 element.split(":")[1].strip()
                                                 for element in
                                                 line.split("-") if element.__contains__(":")}
                                      })
                for item in splits:
                    epoch = item[self.EPOCH]
                    loss = item[self.LOSS]
                    for key in loss:
                        try:
                            output_dict[key]
                        except:
                            output_dict[key] = {}
                        try:
                            fl_loss = float(loss[key])
                            try:
                                output_val = output_dict[key][self.VALUE]
                            except:
                                output_val = 1.0
                            if output_val > fl_loss:
                                output_dict[key] = {self.VALUE: fl_loss,
                                                    self.EPOCH: epoch}
                        except:
                            pass
            print("\n\nLogs filename: ", log_file)
            for key in output_dict:
                try:
                    print("Lowest {} is at {}.".format(key, output_dict[key][self.EPOCH]).capitalize())
                except:
                    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Log parser for Easy Computer Vision Framework')
    parser.add_argument('--filepath', required=False,
                        metavar="path to logs file",
                        help='path to logs file')
    parser.add_argument('--dirpath', required=False,
                        metavar="path to logs directory",
                        help='path to logs directory')
    args = parser.parse_args()

    pl = ParseLogs()
    pl.calculate_best_model(file_path=args.filepath, dir_path=args.dirpath)
