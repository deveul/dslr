#!/usr/bin/python3
# -*-coding:Utf-8 -*

import argparse
import os.path
import pandas as pd
from pandas.api.types import is_numeric_dtype
from matplotlib import pyplot as plt

def is_valid_file(parser, arg):
    if not os.path.exists(arg):
        parser.error("The file {} does not exist!".format(arg))
    elif not arg.endswith('.csv'):
        parser.error("The file {} has no csv extension!".format(arg))
    else:
        return arg

def positive_int_type(arg):
    """ Type function for argparse - a int that must be positive or null """
    try:
        nb = int(arg)
    except ValueError:    
        raise argparse.ArgumentTypeError("Must be an integer")
    if nb < 0 or nb > 100:
        raise argparse.ArgumentTypeError("Argument must be an int, from 1 to 100")
    return nb

class Histogram:
    def __init__(self, args):
        self.bins = args.bins
        self.density = args.density
        self.overlapping = args.overlapping
        self.all_classes = args.all_classes
        self.df = None,
        self.gryf = None
        self.slyt = None
        self.rave = None
        self.huff = None
        self.stack = None
        self.stack_index = 0
        self.fig = plt.figure()
        self.colors = ['r', 'g', 'b', 'y']

    def read_data(self, data_file):
        self.df = pd.read_csv(data_file)
        self.gryf = self.df[self.df['Hogwarts House'] == "Gryffindor"]
        self.slyt = self.df[self.df['Hogwarts House'] == "Slytherin"]
        self.rave = self.df[self.df['Hogwarts House'] == "Ravenclaw"]
        self.huff = self.df[self.df['Hogwarts House'] == "Hufflepuff"]

    def get_stack(self):
        stack = []
        for index, column in enumerate(self.df.columns):
            if is_numeric_dtype(self.df[column].dtypes) and index > 0:
                stack.append(column)
        try:
            self.stack_index = stack.index('Arithmancy')
        except:
            print("Error: it seems that the datas were modified.")
            exit()
        self.stack = stack

    def redraw(self, event): 
        event.canvas.figure.clear()
        y1=self.gryf[self.stack[self.stack_index]]
        y2=self.slyt[self.stack[self.stack_index]]
        y3=self.rave[self.stack[self.stack_index]]
        y4=self.huff[self.stack[self.stack_index]]
        if not self.overlapping:
            event.canvas.figure.gca().hist([y1,y2, y3, y4], self.bins, density = self.density, color=self.colors, label=['Gryffindor', 'Slytherin', 'Ravenclaw', 'Hufflepuff'])
        else:
            event.canvas.figure.gca().hist(y1, self.bins, density=self.density, alpha=0.5, color='r', label='Gryffindor')
            event.canvas.figure.gca().hist(y2, self.bins, density=self.density, alpha=0.5, color='g', label='Slytherin')
            event.canvas.figure.gca().hist(y3, self.bins, density=self.density, alpha=0.5, color='b', label='Ravenclaw')
            event.canvas.figure.gca().hist(y4, self.bins, density=self.density, alpha=0.5, color='yellow', label='Hufflepuff')
        axes = plt.gca()
        axes.set_xlabel("Grades")
        axes.set_ylabel("Count by house")
        plt.title('Grades repartition for {}'.format(self.stack[self.stack_index]))
        plt.legend(loc="upper left")
        event.canvas.draw()

    def press(self, event):
        if event.key != 'q':
            if self.all_classes:
                if event.key == 'right':
                    self.stack_index += 1
                elif event.key == 'left':
                    self.stack_index -= 1
            if event.key == 'o':
                self.overlapping = not self.overlapping
            elif event.key == 'd':
                self.density = not self.density
            elif event.key == 'a':
                self.all_classes = not self.all_classes
            if self.stack_index > len(self.stack) - 1:
                self.stack_index = 0
            elif self.stack_index < 0:
                self.stack_index = len(self.stack) - 1
        
            self.redraw(event)

    def display_histogram(self):
        y1=self.gryf[self.stack[self.stack_index]]
        y2=self.slyt[self.stack[self.stack_index]]
        y3=self.rave[self.stack[self.stack_index]]
        y4=self.huff[self.stack[self.stack_index]]
        if not self.overlapping:
            plt.hist([y1,y2, y3, y4], self.bins, density = self.density, color=self.colors, label=['Gryffindor', 'Slytherin', 'Ravenclaw', 'Hufflepuff'])
        else:
            plt.hist(y1, self.bins, density=self.density, alpha=0.5, color='r', label='Gryffindor')
            plt.hist(y2, self.bins, density=self.density, alpha=0.5, color='g', label='Slytherin')
            plt.hist(y3, self.bins, density=self.density, alpha=0.5, color='b', label='Ravenclaw')
            plt.hist(y4, self.bins, density=self.density, alpha=0.5, color='yellow', label='Hufflepuff')
        axes = plt.gca()
        axes.set_xlabel("Grades")
        axes.set_ylabel("Count by house")
        plt.title('Grades repartition for {}'.format(self.stack[self.stack_index]))
        plt.legend(loc="upper left")

        self.fig.canvas.mpl_connect('key_press_event', self.press)
    
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_file", help="the csv file containing the data set", type=lambda x: is_valid_file(parser, x))
    parser.add_argument("-a", "--all_classes", help="display an histogram for each course", action="store_true")
    parser.add_argument("-o", "--overlapping", help="display the houses on top of each other", action="store_true")
    parser.add_argument("-d", "--density", help="y axis in percentage instead of number of students", action="store_true")
    parser.add_argument("-b", "--bins", help="Number of bins (intervals) per house", type=positive_int_type, default=10)
    args = parser.parse_args()
    histogram = Histogram(args)
    histogram.read_data(args.data_file)
    histogram.get_stack()
    histogram.display_histogram()

if __name__ == "__main__":
    main()