import pickle as pk



with open('./7-uMJ_5WsZM', "rb") as fp:   # Unpickling
    fileX = pk.load(fp)
    #for x in fileX:
#    print(len(fileX))
#    print(len(fileX[0]))

#    for frameNum in range(len(fileX)):
    print("rois")
    for i in range(len(fileX)):
        print(fileX[i]['rois'])
#    print("scores")
#    print(fileX[10]['scores'])
#        print(fileX[frameNum]['rois'])
#        print(fileX[frameNum]['centers'])
#        print(fileX[frameNum]['areas'])
#        print(fileX[frameNum]['frameSize_w_h'])


