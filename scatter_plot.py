#!/usr/bin/python3
# -*-coding:Utf-8 -*

import argparse
import os.path
import pandas as pd
from pandas.api.types import is_numeric_dtype
from matplotlib import pyplot as plt
from utils.stats_functions import pearson_core

def is_valid_file(parser, arg):
    if not os.path.exists(arg):
        parser.error("The file {} does not exist!".format(arg))
    elif not arg.endswith('.csv'):
        parser.error("The file {} has no csv extension!".format(arg))
    else:
        return arg

def ft_abs(nb):
    if nb < 0:
        return -1 * nb
    return nb

class ScatterPlot:
    def __init__(self):
        self.df = None
        self.scores = None
        self.labels = None
        self.stack = None
        self.stack_index = 0
        self.best = False
        self.index_best = None
        self.all = True
        self.fig = plt.figure()

    def read_data(self, data_file):
        df = pd.read_csv(data_file)
        scores = []
        labels = []
        stack = []
        best = 0
        df = df.drop(columns="Index")
        for column in df.columns:
            if is_numeric_dtype(df[column].dtypes):
                for column2 in df.columns:
                    if is_numeric_dtype(df[column2].dtypes) and column != column2:
                        if "{}-{}".format(column2, column) not in labels:
                            score = pearson_core(df[column], df[column2])
                            if ft_abs(score) > ft_abs(best):
                                best = score
                                self.index_best = len(scores)
                            scores.append(score)
                            labels.append("{}-{}".format(column, column2))
                            stack.append([column, column2])

        self.df = df
        self.scores = scores
        self.labels = labels
        self.stack = stack
    
    def redraw(self, event):
        event.canvas.figure.clear()
        if self.all:
            axes = plt.gca()
            axes.set_ylim(-1.25, 1.25)
            plt.axhline(1, color='grey')
            plt.axhline(-1, color='grey')
            x = [index for index, _ in enumerate(self.scores)]
            plt.xticks(x, self.labels, rotation="vertical")
            plt.scatter(x, self.scores, c='darkgreen', alpha=1, label='Students')
            plt.title('Correlated features')
            plt.legend(loc="upper right")
            self.fig.subplots_adjust(bottom=0.5)
        elif self.best:
            axes = plt.gca()
            axes.set_xlabel(self.stack[self.index_best][0])
            axes.set_ylabel(self.stack[self.index_best][1])
            plt.scatter([x for x in self.df[self.stack[self.index_best][0]]], [y for y in self.df[self.stack[self.index_best][1]]], c='blue', alpha=0.5, label='Students')
            plt.title('Best Correlation: {:0.2f}'.format(self.scores[self.index_best]))
            plt.legend(loc="upper right")
            self.fig.subplots_adjust(bottom=0.15)
        else:
            axes = plt.gca()
            axes.set_xlabel(self.stack[self.stack_index][0])
            axes.set_ylabel(self.stack[self.stack_index][1])
            plt.scatter([x for x in self.df[self.stack[self.stack_index][0]]], [y for y in self.df[self.stack[self.stack_index][1]]], c='blue', alpha=0.5, label='Students')
            plt.title('Correlation: {:0.2f}'.format(self.scores[self.stack_index]))
            plt.legend(loc="upper right")
            self.fig.subplots_adjust(bottom=0.15)
        event.canvas.draw()

    def press(self, event):
        key = event.key
        if key != 'q':
            if key == 'right' or key == 'left' :
                if self.all == True:
                    self.all = not self.all
                elif self.best == True:
                    self.best = not self.best
                    self.stack_index = self.index_best
                if key == 'right':
                    self.stack_index += 1
                elif key == 'left':
                    self.stack_index -= 1
            elif key == 'a':
                self.all = not self.all
            elif key =='b':
                self.all = False
                self.best = not self.best
            if self.stack_index > len(self.stack) - 1:
                self.stack_index = 0
            elif self.stack_index < 0:
                self.stack_index = len(self.stack) - 1

            self.redraw(event)

    def display_correlations(self):
        axes = plt.gca()
        axes.set_ylim(-1.25, 1.25)
        plt.axhline(1, color='grey')
        plt.axhline(-1, color='grey')
        x = [index for index, _ in enumerate(self.scores)]
        plt.xticks(x, self.labels, rotation="vertical")
        plt.scatter(x, self.scores, c='darkgreen', alpha=1, label='Students')
        plt.title('Correlated features')
        plt.legend(loc="upper right")
        self.fig.subplots_adjust(bottom=0.5)

        self.fig.canvas.mpl_connect('key_press_event', self.press)

        plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_file", help="the csv file containing the data set", type=lambda x: is_valid_file(parser, x))
    args = parser.parse_args()
    scatter_plot = ScatterPlot()
    scatter_plot.read_data(args.data_file)
    scatter_plot.display_correlations()

if __name__ == "__main__":
    main()