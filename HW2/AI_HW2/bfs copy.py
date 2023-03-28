import csv
edgeFile = 'edges.csv'





def bfs(start, end):
    # Begin your code (Part 1)
    #raise NotImplementedError("To be implemented")
    data = []
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
    
    bfs_q = []
    cur_addr = start
    for r in all_rows:
        if((r[0] == cur_addr or r[1]==cur_addr)and r[4] == 0):
            if(r[0]==start):
                r.append(int(1)) #r[5]: r[0 or 1] is the current location
            else: 
                r.append(int(0))
            r[4] = 1
            bfs_q.append(r)
        
    dest = []
    while(len(bfs_q)!=0):
        
        cur = bfs_q[0]
        visited = visited+1
        if (cur[5] == 1):
            location = cur[1]
        else :
            location = cur[0]
        #else:
        for r in all_rows:
            if(r[0] == location and r[4] == 0):
                r[4] = cur[4]+1
                r.append(int(1))
                bfs_q.append(r)
                if(r[1]==end):
                    num_visited = visited
                    dest = r
                    break
            elif(r[1] == location and r[4] == 0):
                r[4] = cur[4]+1
                r.append(int(0))
                bfs_q.append(r)
                if(r[0]==end):
                    num_visited = visited
                    dest = r
                    break
        bfs_q.pop(0)



    dict_row = dict()
    for i in range(dest[4]+1):
        dict_row[i] = []
    

    
    for r in all_rows:
        if(r[4] in dict_row):
            dict_row[r[4]].append(r)
    #print (dict_row)
    
    dist = 0
    path = [start]
    route = [dest]
    cur_addr = dest
    for i in range(dest[4]-1):
        #print('-------------')
        for val in dict_row[dest[4]-1-i]:
            if(val[val[5]]==cur_addr[abs(1-cur_addr[5])]):
                route.append(val)
                cur_addr = val
                break
    
    route.reverse()
    for r in route:
        path.append(r[r[5]])
        dist = dist + r[2]

    print(dist)
    print (path)

    return path,dist,num_visited
    # End your code (Part 1)

if __name__ == '__main__':
    path, dist, num_visited = bfs(2270143902, 1079387396)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total distance of path: {dist}')
    print(f'The number of visited nodes: {num_visited}')
