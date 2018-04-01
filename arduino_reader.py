import os.path as osp
from SerialReader import SerialReader
import sys

FOLDER_SPACE = "/home/sabyasachi/Documents/Indiana_University/Spring_2018/Applied_Machine_Learning/Final_Project/test/"


def start():
    while True:
        cmd = input('Enter T for train , type anything else to exit : ')
        processCmd(cmd)


def train():
    word = input('Enter the word for training : ')
    while True:
        cmd = input('Press enter to start or "e" to exit : ')
        if cmd == '':
            rdr = SerialReader(1, "Reader-Thread", 1)
            rdr.start()
            input('Press enter to stop : ')
            data = rdr.stop()
            storeData(word, data)
        elif cmd == 'e':
            break;


def processCmd(cmd):
    if cmd.upper() == 'T':
        train()
    else:
        print('Exiting ...  ')
        sys.exit()          


def storeData(word, data):
    filename = "_".join(word.split()) 
    file = open(FOLDER_SPACE + osp.sep + filename + '.txt', 'a+', 1024)
    for dp in data:
        file.write(dp.decode('utf-8', 'ignore'))
    file.write("---------END--------\n")
    file.close()


if __name__ == "__main__":
    start()
