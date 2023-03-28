import csv
edgeFile = 'edges.csv'
dest = []
e = int
find = False



def recur(parent,all_rows):
    if(find == False):
        for r in all_rows :
            if(r[1]==e):
                find == True
                print("find__________________________________________")
                dest = r
                break
            elif(r[0]==parent[1] and r[4]==0):
                r[4] = parent[4]+1
                recur(r,all_rows)


def dfs(start, end):
    # Begin your code (Part 2)
    #raise NotImplementedError("To be implemented")
    e = end
    visited=0
    with open(edgeFile, newline='') as file:
        d = csv.reader(file)
        all_rows = list(d)
        all_rows.pop(0)
        for r in all_rows:
            r[0] = int(r[0])
            r[1] = int(r[1])
            r[2] = float(r[2])
            r.append(int(0)) 


    cur_addr = start
    for r in all_rows:
        if((r[0] == cur_addr)and r[4] == 0):
            visited = visited+1
            r[4] = 1
            recur(r,all_rows)
    print (dest)
    # End your code (Part 2)


if __name__ == '__main__':
    path, dist, num_visited = dfs(2270143902, 1079387396)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total distance of path: {dist}')
    print(f'The number of visited nodes: {num_visited}')
