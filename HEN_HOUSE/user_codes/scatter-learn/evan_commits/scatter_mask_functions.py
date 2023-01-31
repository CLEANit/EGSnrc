def slabs(xl,ml,n,i,j):
    xi = xl[i]
    mi = ml[i]
    return bool((xi//mi)%n)

def columns(xl,ml,n,i,j):
    ## i,j switched
    return bool(slabs(xl,ml,n,i,j) == slabs(xl,ml,n,j,i))

def checkers(xl,ml,n,i,j):
    ## j ignored since all three dims are used.
    return bool( bool(slabs(xl,ml,n,0,0) == slabs(xl,ml,n,1,1)) == slabs(xl,ml,n,2,2) )


    
