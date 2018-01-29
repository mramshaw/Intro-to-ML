#!/usr/bin/env python

import sys

def main(argv):
    for arg in argv:
        try:
            __import__(arg)
            print('"%s" was imported' % (arg))
        except ImportError:
            print('"%s" could not be imported - try "pip install --user %s"' % (arg, arg))

if __name__ == "__main__":
    main(sys.argv[1:])
