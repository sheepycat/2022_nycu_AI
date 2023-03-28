import csv
edgeFile = 'edges.csv'
heuristicFile = 'heuristic.csv'
from queue import PriorityQueue


def astar(start, end):
    # Begin your code (Part 4)
    #raise NotImplementedError("To be implemented")
    # End your code (Part 4)
    visited=0
    with open(edgeFile, newline='') as file: # open edge data
        d = csv.reader(file)
        all_rows = list(d) #store data into an array
        all_rows.pop(0) #get rid of the first line 
        for r in all_rows:
            r[0] = int(r[0]) # each row : [start,end,distance,speed limit,found @ which round, parent start, parent end, h]
            r[1] = int(r[1])
            r[2] = float(r[2])
            r.append(int(0)) 
            r.append(int(0))
            r.append(int(0))
            r.append(int(0))
    
    node_distance = [] # store h of each node, h depends on end
    if(end == 1079387396) : key = 1
    elif (end == 1737223506) : key = 2
    elif(end == 8513026827) : key = 3

    with open(heuristicFile, newline='') as file: # read heuristic data
        d = csv.reader(file)
        n = list(d)
        n.pop(0)
        for row in n:
            node_distance.append([int(row[0]),float(row[key])]) # store the data into an array
    
    for r in all_rows:
        for n in node_distance: #update h data of edges in all_rows
            if(n[0]==r[1]):
                r[7] = n[1]
                break

    
    bfs_q = PriorityQueue()
    for r in all_rows:
        if((r[0] == start)and r[4] == 0): # put the start edge into the queue
            visited = visited+1
            r[4] = 1
            bfs_q.put([(r[2]+r[7]),r]) #priority : distance(cost) + h of the edge
    
        
    dest = []
    find = False
    while( (not bfs_q.empty()) and find == False): # while not find or queue not empty
        
        c = bfs_q.get() # get the first element in the queue (with highest priority)
        cur = c[1]
        location = cur[1] # get current location
        

        for r in all_rows:
            if(r[0] == location and r[4] == 0): # if start of r = current edge's end，and and hasn't been discovered
                visited = visited+1 #visit node +1
                r[4] = cur[4]+1 # round = parent's round+1
                r[5] = cur[0] # update parent's address
                r[6] = cur[1]
                bfs_q.put([(c[0]-cur[7]+r[2]+r[7]),r]) #priority :  c[0] = cummulated cost,cur[7]=parent's h, r[2] = cost, r[7] = h(x)
                if(r[1]==end): # when find end
                    num_visited = visited
                    dest = r #destination = r
                    find = True
                    break

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

if __name__ == '__main__':
    path, dist, num_visited = astar(2270143902, 1079387396)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total distance of path: {dist}')
    print(f'The number of visited nodes: {num_visited}')
