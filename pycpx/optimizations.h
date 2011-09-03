#ifndef OPTIMIZATIONS_H
#define OPTIMIZATIONS_H

/* Branching hints -- unlikely and likely -- and expected values (e.g. switch statements). */

#ifdef __GNUC__
/* Test for GCC > 2.95 */
#if __GNUC__ > 2 || (__GNUC__ == 2 && (__GNUC_MINOR__ > 95)) 

#define likely(x)   __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)
#define likely_value(x, y) __builtin_expect((x), (y))

#else /* __GNUC__ > 2 ... */

#define likely(x)   (x)
#define unlikely(x) (x)
#define likely_value(x, y)

#endif /* __GNUC__ > 2 ... */

#else /* __GNUC__ */

#define likely(x)   (x)
#define unlikely(x) (x)
#define likely_value(x, y)

#endif /* __GNUC__ */

/* Prefetching operations */ 

/* Test for GCC >= 3.2 */
#if __GNUC__ > 3 || (__GNUC__ == 3 && (__GNUC_MINOR__ >= 2))

#define prefetch_ro(addr)  __builtin_prefetch((addr), 0)
#define prefetch_rw(addr)  __builtin_prefetch((addr), 1)
#define prefetch_ro_keep(addr)  __builtin_prefetch((addr), 0)
#define prefetch_rw_keep(addr)  __builtin_prefetch((addr), 1)

#ifndef PREFETCH_STRIDE
#define PREFETCH_STRIDE (4*32)  // from linux kernel, most common size.
#endif

#define ON_STRIDE(x) ( unlikely( ((size_t(x)) & 0x007F) == 0) ) 

template <typename T> static inline void prefetch_range_ro(const T *const addr, size_t len)
{
    char *end = (char*)(addr + len);

    for (char *cp = (char*)addr; cp < end; cp += PREFETCH_STRIDE)
	prefetch_ro(cp);
}

template <typename T> static inline void prefetch_range_rw(const T *const addr, size_t len)
{
    char *end = (char*)(addr + len);

    for (char *cp = (char*)addr; cp < end; cp += PREFETCH_STRIDE)
	prefetch_rw(cp);
}

#else

#define prefetch_ro(addr)  
#define prefetch_rw(addr)  
#define prefetch_ro_keep(addr)  
#define prefetch_rw_keep(addr)  

template <typename T> static inline void prefetch_range_ro(const T const*, size_t){;}
template <typename T> static inline void prefetch_range_rw(const T const*, size_t){;}

#ifndef PREFETCH_STRIDE
#define PREFETCH_STRIDE 8
#endif

#define ON_STRIDE(x) ( unlikely( ((size_t(x)) & 0x007F) == 0) ) 

#warning "prefetch not defined."

#endif

#endif /* OPTIMIZATIONS_H */
