from numpy import dtype

npdouble_dt = dtype('double')
npfloat_dt = dtype('float')
npint_dt = dtype('int')
npuint_dt = dtype('uint')

if sizeof(long) == 4:
    nplong_dt = dtype('int32')
    npulong_dt = dtype('uint32')
elif sizeof(long) == 8:
    nplong_dt = dtype('int64')
    npulong_dt = dtype('uint64')
else:
    raise TypeError("Type Conversion table needs updating (long)")

if sizeof(size_t) == 8:
    npsize_t_dt = dtype('uint64')
elif sizeof(size_t) == 4:
    npsize_t_dt = dtype('uint32')
else:
    raise TypeError("Type Conversion table needs updating (size_t)")

npdouble = npdouble_dt.type
npfloat = npfloat_dt.type
npint = npint_dt.type
npuint = npuint_dt.type
nplong = nplong_dt.type
npulong = npulong_dt.type
npsize_t = npsize_t_dt.type
