
import sys
import os
import subprocess


class ParseLogs():

    TEE = "tee"

    @staticmethod
    def redirect_logs_to_file(logs_path):
        tee = subprocess.Popen([ParseLogs.TEE, logs_path], stdin=subprocess.PIPE)
        os.dup2(tee.stdin.fileno(), sys.stdout.fileno())
        os.dup2(tee.stdin.fileno(), sys.stderr.fileno())

