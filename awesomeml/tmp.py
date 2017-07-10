#!/usr/bin/env python
# encoding: utf-8

"""
@description: 

@author: BaoQiang
@time: 2017/7/7 21:10
"""

root_path = 'C:\\Users\\BaoQiang\\Desktop\\'


def run():
    with open('{}/1.txt'.format(root_path), 'r', encoding='utf-8') as f, \
            open('{}/2.txt'.format(root_path), 'w', encoding='utf-8') as fw:
        for line in f:
            line = line.strip()

            attr = line.split('\t')

            if len(attr) < 3:
                continue

            key = attr[1]

            value = '|'.join(attr[2:])

            fw.write('{}={}\n'.format(key, value))


def main():
    run()


if __name__ == '__main__':
    main()
