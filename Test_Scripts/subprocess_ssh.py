import subprocess
import time


for i in range(1000):

#if True:

    bshCmd = "ssh marius@pymonnto.ddns.net 'ls' -tt"
    #bshCmd = "scp C:/Users/Nutzer/Programmieren/Python_Modular_Neural_Network_Toolbox/SequenceSORN/Test_Scripts/subprocess_ssh.py marius@pymonnto.ddns.net:subprocess_ssh.py"

    #process = subprocess.Popen(bshCmd.split(), stdout=subprocess.PIPE, check=True)
    #output = process.communicate()[0]
    #for o in output.decode("utf-8").split('\n'):
    #    print(o)
    #process.terminate()

    result = subprocess.run(["ssh", "marius@pymonnto.ddns.net", "ls"],
                           shell=False,
                           stdin=None,
                           stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE,
                           check=True).stdout.decode()

    print(result)

    #print(output.decode("utf-8"))
