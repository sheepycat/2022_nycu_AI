import csv
edgeFile = 'edges.csv'





def bfs(start, end):
    # Begin your code (Part 1)
    #raise NotImplementedError("To be implemented")
    visited=0
    with open(edgeFile, newline='') as file:# open edge data
        d = csv.reader(file) 
        all_rows = list(d) #store data into an array
        all_rows.pop(0) #get rid of the first line 
        for r in all_rows:
            r[0] = int(r[0]) # each row : [start,end,distance,speed limit,found @ which round, parent start, parent end]
            r[1] = int(r[1])
            r[2] = float(r[2])
            r.append(int(0))
            r.append(int(0))
            r.append(int(0)) 
    
    bfs_q = []
    cur_addr = start
    for r in all_rows: # put the start edge into the list
        if((r[0] == cur_addr)and r[4] == 0):
            visited = visited+1
            r[4] = 1
            bfs_q.append(r)
        
    dest = []
    find = False
    while(len(bfs_q)!=0 and find == False): # while not find or queue not empty
        cur = bfs_q[0] # get the first element in the list
        location = cur[1]
        
        for r in all_rows:
            if(r[0] == location and r[4] == 0): # if start of r = current edge's end，and hasn't been discovered
                visited = visited+1 #visit node +1
                r[4] = cur[4]+1 # round = parent's round+1
                r[5] = cur[0] # update parent's address
                r[6] = cur[1]
                bfs_q.append(r) # append r to the last of list
                if(r[1]==end): # when find end
                    num_visited = visited
                    dest = r #destination = r
                    find = True
                    break
        bfs_q.pop(0) # pop first

    d = [dest] # store path edges
    curr = dest
    while (curr[0]!=start):
        for r in all_rows:
            if(r[0]==curr[5] and r[1]== curr[6]): # start from destination, find parents iteratively until reach start
                d.append(r)
                curr = r
                break
    d.reverse()
    path = [start]
    dist = 0
    for r in d:
        path.append(r[1]) # store nodes of edge in sequence
        dist = dist + r[2] # sum distance

    return path,dist,num_visited
    # End your code (Part 1)

if __name__ == '__main__':
    path, dist, num_visited = bfs(2270143902, 1079387396)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total distance of path: {dist}')
    print(f'The number of visited nodes: {num_visited}')
