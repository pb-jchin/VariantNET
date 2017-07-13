import sys


for row in sys.stdin:
    row = row.strip().split()
    if row[0][0] == "#":
        continue
    last = row[-1]
    het_hom = last.split(":")[0].replace("/","|").split("|")
    p1, p2 = het_hom
    p1 = int(p1)
    p2 = int(p2)
    p1, p2 = (p1, p2) if p1 < p2 else (p2, p1)
    print row[1], row[3], row[4], p1, p2
