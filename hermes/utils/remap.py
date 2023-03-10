def rescale_2d_data_linear(x1,y1,x2):
    # linear interopolation
    y2 = list()
    if min(x2) < min(x1):
        raise Exception("Out of range min(x2) < min(x1)")
    if max(x2) > max(x1):
        raise Exception("Out of range max(x2) > max(x1)")
    if x1 != sorted(x1):
        raise Exception("input list x1 is not sorted")
    if x2 != sorted(x2):
        raise Exception("input list x2 is not sorted")
    
    id_x = 0;
    for x in x2:
        y=None
        x_p0 = x1[id_x]
        if x == x_p0:
            # perfect match
            y = y1[id_x]
        else:
            x_p1 = x1[id_x + 1]
            while x_p1 < x:
                #print('next',x_p0,x,x_p1)
                id_x = id_x + 1
                x_p0 = x1[id_x]
                x_p1 = x1[id_x + 1]
            if x == x_p0:
                # perfect match
                y = y1[id_x]
            elif x == x_p1:
                # perfect match
                y = y1[id_x + 1]
            elif x_p0 < x and x < x_p1:
                # interpolate
                y_p0 = y1[id_x]
                y_p1 = y1[id_x + 1]
                y = (x - x_p0) * (y_p1 - y_p0) / (x_p1 - x_p0) + y_p0
            else:
                raise Exception("Something went wrong during interpolation")
        if y:
            y2.append(y)
        else:
            raise Exception("Something went wrong during rescale at location:", x)
    
    if len(x2) != len(y2):
        raise Exception("Something went wrong lists not equal")
    
    return y2