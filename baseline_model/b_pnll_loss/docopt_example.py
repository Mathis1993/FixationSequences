"""
Example program.

Usage:
  example command [<cmd_arg>]...
  example [-br] -p=<opt_arg> <argument>
  example -h | --help
  example --version

 Options:
   -h, --help       Show this message.
   -b, --beer       Drink beer.
   -r, --rock       Play AC/DC.
   -p, --pub=<p>    Which pub.
   --version        Print the version.
"""

from docopt import docopt
from pprint import pprint

if __name__ == '__main__':
    arguments = docopt(__doc__, version='FIXME')
    pprint(arguments)

print("success")