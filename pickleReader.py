import pickle as pk



with open('./7-uMJ_5WsZM', "rb") as fp:   # Unpickling
    fileX = pk.load(fp)
    #for x in fileX:
#    print(len(fileX))
#    print(len(fileX[0]))

#    for frameNum in range(len(fileX)):
    for i in range(len(fileX)):
#        print(fileX[i]['centers'])
        print(fileX[i]['areas'])


