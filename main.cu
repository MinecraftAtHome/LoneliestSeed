#include <string.h>
#include <stdio.h>
#include <inttypes.h>
#include <math.h>

#ifndef RNG_H_
#define RNG_H_

#define __STDC_FORMAT_MACROS 1

#include <stdlib.h>
#include <stddef.h>
#include <inttypes.h>

#ifdef BOINC
  #include "boinc_api.h"
#if defined _WIN32 || defined _WIN64
  #include "boinc_win.h"
#endif
#endif

///=============================================================================
///                      Compiler and Platform Features
///=============================================================================

typedef int8_t      i8;
typedef uint8_t     u8;
typedef int16_t     i16;
typedef uint16_t    u16;
typedef int32_t     i32;
typedef uint32_t    u32;
typedef int64_t     i64;
typedef uint64_t    u64;
typedef float       f32;
typedef double      f64;


#define STRUCT(S) typedef struct S S; struct S

#if __GNUC__

#define IABS(X)                 __builtin_abs(X)
#define PREFETCH(PTR,RW,LOC)    __builtin_prefetch(PTR,RW,LOC)
#define likely(COND)            (__builtin_expect(!!(COND),1))
#define unlikely(COND)          (__builtin_expect((COND),0))
#define ATTR(...)               __attribute__((__VA_ARGS__))
#define BSWAP32(X)              __builtin_bswap32(X)
#define UNREACHABLE()           __builtin_unreachable()

#else

#define IABS(X)                 ((int)abs(X))
#define PREFETCH(PTR,RW,LOC)
#define likely(COND)            (COND)
#define unlikely(COND)          (COND)
#define ATTR(...)
__device__ __host__ static inline uint32_t BSWAP32(uint32_t x) {
    x = ((x & 0x000000ff) << 24) | ((x & 0x0000ff00) <<  8) |
        ((x & 0x00ff0000) >>  8) | ((x & 0xff000000) >> 24);
    return x;
}
#if _MSC_VER
#define UNREACHABLE()           
#else
#define UNREACHABLE()           exit(1) // [[noreturn]]
#endif

#endif

/// imitate amd64/x64 rotate instructions

__device__ __host__ static inline ATTR(const, always_inline, artificial)
uint64_t rotl64(uint64_t x, uint8_t b)
{
    return (x << b) | (x >> (64-b));
}

__device__ __host__ static inline ATTR(const, always_inline, artificial)
uint32_t rotr32(uint32_t a, uint8_t b)
{
    return (a >> b) | (a << (32-b));
}

/// integer floor divide
__device__ __host__ static inline ATTR(const, always_inline)
int32_t floordiv(int32_t a, int32_t b)
{
    int32_t q = a / b;
    int32_t r = a % b;
    return q - ((a ^ b) < 0 && !!r);
}

///=============================================================================
///                    C implementation of Java Random
///=============================================================================

__device__ __host__ static inline void setSeed(uint64_t *seed, uint64_t value)
{
    *seed = (value ^ 0x5deece66d) & ((1ULL << 48) - 1);
}

__device__ __host__ static inline int next(uint64_t *seed, const int bits)
{
    *seed = (*seed * 0x5deece66d + 0xb) & ((1ULL << 48) - 1);
    return (int) ((int64_t)*seed >> (48 - bits));
}

__device__ __host__ static inline int nextInt(uint64_t *seed, const int n)
{
    int bits, val;
    const int m = n - 1;

    if ((m & n) == 0) {
        uint64_t x = n * (uint64_t)next(seed, 31);
        return (int) ((int64_t) x >> 31);
    }

    do {
        bits = next(seed, 31);
        val = bits % n;
    }
    while (bits - val + m < 0);
    return val;
}

__device__ __host__ static inline uint64_t nextLong(uint64_t *seed)
{
    return ((uint64_t) next(seed, 32) << 32) + next(seed, 32);
}

__device__ __host__ static inline float nextFloat(uint64_t *seed)
{
    return next(seed, 24) / (float) (1 << 24);
}

__device__ __host__ static inline double nextDouble(uint64_t *seed)
{
    uint64_t x = (uint64_t)next(seed, 26);
    x <<= 27;
    x += next(seed, 27);
    return (int64_t) x / (double) (1ULL << 53);
}

/* A macro to generate the ideal assembly for X = nextInt(*S, 24)
 * This is a macro and not an inline function, as many compilers can make use
 * of the additional optimisation passes for the surrounding code.
 */
#define JAVA_NEXT_INT24(S,X)                \
    do {                                    \
        uint64_t a = (1ULL << 48) - 1;      \
        uint64_t c = 0x5deece66dULL * (S);  \
        c += 11; a &= c;                    \
        (S) = a;                            \
        a = (uint64_t) ((int64_t)a >> 17);  \
        c = 0xaaaaaaab * a;                 \
        c = (uint64_t) ((int64_t)c >> 36);  \
        (X) = (int)a - (int)(c << 3) * 3;   \
    } while (0)


/* Jumps forwards in the random number sequence by simulating 'n' calls to next.
 */
__device__ __host__ static inline void skipNextN(uint64_t *seed, uint64_t n)
{
    uint64_t m = 1;
    uint64_t a = 0;
    uint64_t im = 0x5deece66dULL;
    uint64_t ia = 0xb;
    uint64_t k;

    for (k = n; k; k >>= 1)
    {
        if (k & 1)
        {
            m *= im;
            a = im * a + ia;
        }
        ia = (im + 1) * ia;
        im *= im;
    }

    *seed = *seed * m + a;
    *seed &= 0xffffffffffffULL;
}


///=============================================================================
///                               Xoroshiro 128
///=============================================================================

STRUCT(Xoroshiro)
{
    uint64_t lo, hi;
};

__device__ __host__ static inline void xSetSeed(Xoroshiro *xr, uint64_t value)
{
    const uint64_t XL = 0x9e3779b97f4a7c15ULL;
    const uint64_t XH = 0x6a09e667f3bcc909ULL;
    const uint64_t A = 0xbf58476d1ce4e5b9ULL;
    const uint64_t B = 0x94d049bb133111ebULL;
    uint64_t l = value ^ XH;
    uint64_t h = l + XL;
    l = (l ^ (l >> 30)) * A;
    h = (h ^ (h >> 30)) * A;
    l = (l ^ (l >> 27)) * B;
    h = (h ^ (h >> 27)) * B;
    l = l ^ (l >> 31);
    h = h ^ (h >> 31);
    xr->lo = l;
    xr->hi = h;
}

__device__ __host__ static inline uint64_t xNextLong(Xoroshiro *xr)
{
    uint64_t l = xr->lo;
    uint64_t h = xr->hi;
    uint64_t n = rotl64(l + h, 17) + l;
    h ^= l;
    xr->lo = rotl64(l, 49) ^ h ^ (h << 21);
    xr->hi = rotl64(h, 28);
    return n;
}

__device__ __host__ static inline int xNextInt(Xoroshiro *xr, uint32_t n)
{
    uint64_t r = (xNextLong(xr) & 0xFFFFFFFF) * n;
    if ((uint32_t)r < n)
    {
        while ((uint32_t)r < (~n + 1) % n)
        {
            r = (xNextLong(xr) & 0xFFFFFFFF) * n;
        }
    }
    return r >> 32;
}

__device__ __host__ static inline double xNextDouble(Xoroshiro *xr)
{
    return (xNextLong(xr) >> (64-53)) * 1.1102230246251565E-16;
}

__device__ __host__ static inline float xNextFloat(Xoroshiro *xr)
{
    return (xNextLong(xr) >> (64-24)) * 5.9604645E-8F;
}

__device__ __host__ static inline void xSkipN(Xoroshiro *xr, int count)
{
    while (count --> 0)
        xNextLong(xr);
}

__device__ __host__ static inline uint64_t xNextLongJ(Xoroshiro *xr)
{
    int32_t a = xNextLong(xr) >> 32;
    int32_t b = xNextLong(xr) >> 32;
    return ((uint64_t)a << 32) + b;
}

__device__ __host__ static inline int xNextIntJ(Xoroshiro *xr, uint32_t n)
{
    int bits, val;
    const int m = n - 1;

    if ((m & n) == 0) {
        uint64_t x = n * (xNextLong(xr) >> 33);
        return (int) ((int64_t) x >> 31);
    }

    do {
        bits = (xNextLong(xr) >> 33);
        val = bits % n;
    }
    while (bits - val + m < 0);
    return val;
}


//==============================================================================
//                              MC Seed Helpers
//==============================================================================

/**
 * The seed pipeline:
 *
 * getLayerSalt(n)                -> layerSalt (ls)
 * layerSalt (ls), worldSeed (ws) -> startSalt (st), startSeed (ss)
 * startSeed (ss), coords (x,z)   -> chunkSeed (cs)
 *
 * The chunkSeed alone is enough to generate the first PRNG integer with:
 *   mcFirstInt(cs, mod)
 * subsequent PRNG integers are generated by stepping the chunkSeed forwards,
 * salted with startSalt:
 *   cs_next = mcStepSeed(cs, st)
 */

__device__ __host__ static inline uint64_t mcStepSeed(uint64_t s, uint64_t salt)
{
    return s * (s * 6364136223846793005ULL + 1442695040888963407ULL) + salt;
}

__device__ __host__ static inline int mcFirstInt(uint64_t s, int mod)
{
    int ret = (int)(((int64_t)s >> 24) % mod);
    if (ret < 0)
        ret += mod;
    return ret;
}

__device__ __host__ static inline int mcFirstIsZero(uint64_t s, int mod)
{
    return (int)(((int64_t)s >> 24) % mod) == 0;
}

__device__ __host__ static inline uint64_t getChunkSeed(uint64_t ss, int x, int z)
{
    uint64_t cs = ss + x;
    cs = mcStepSeed(cs, z);
    cs = mcStepSeed(cs, x);
    cs = mcStepSeed(cs, z);
    return cs;
}

__device__ __host__ static inline uint64_t getLayerSalt(uint64_t salt)
{
    uint64_t ls = mcStepSeed(salt, salt);
    ls = mcStepSeed(ls, salt);
    ls = mcStepSeed(ls, salt);
    return ls;
}

__device__ __host__ static inline uint64_t getStartSalt(uint64_t ws, uint64_t ls)
{
    uint64_t st = ws;
    st = mcStepSeed(st, ls);
    st = mcStepSeed(st, ls);
    st = mcStepSeed(st, ls);
    return st;
}

__device__ __host__ static inline uint64_t getStartSeed(uint64_t ws, uint64_t ls)
{
    uint64_t ss = ws;
    ss = getStartSalt(ss, ls);
    ss = mcStepSeed(ss, 0);
    return ss;
}


///============================================================================
///                               Arithmatic
///============================================================================


/* Linear interpolations
 */
__device__ __host__ static inline double lerp(double part, double from, double to)
{
    return from + part * (to - from);
}

__device__ __host__ static inline double lerp2(
        double dx, double dy, double v00, double v10, double v01, double v11)
{
    return lerp(dy, lerp(dx, v00, v10), lerp(dx, v01, v11));
}

__device__ __host__ static inline double lerp3(
        double dx, double dy, double dz,
        double v000, double v100, double v010, double v110,
        double v001, double v101, double v011, double v111)
{
    v000 = lerp2(dx, dy, v000, v100, v010, v110);
    v001 = lerp2(dx, dy, v001, v101, v011, v111);
    return lerp(dz, v000, v001);
}

__device__ __host__ static inline double clampedLerp(double part, double from, double to)
{
    if (part <= 0) return from;
    if (part >= 1) return to;
    return lerp(part, from, to);
}

/* Find the modular inverse: (1/x) | mod m.
 * Assumes x and m are positive (less than 2^63), co-prime.
 */
__device__ __host__ static inline ATTR(const)
uint64_t mulInv(uint64_t x, uint64_t m)
{
    uint64_t t, q, a, b, n;
    if ((int64_t)m <= 1)
        return 0; // no solution

    n = m;
    a = 0; b = 1;

    while ((int64_t)x > 1)
    {
        if (m == 0)
            return 0; // x and m are co-prime
        q = x / m;
        t = m; m = x % m;     x = t;
        t = a; a = b - q * a; b = t;
    }

    if ((int64_t)b < 0)
        b += n;
    return b;
}


#endif /* RNG_H_ */

STRUCT(Range)
{
    int scale;
    int x, z, sx, sz;
    int y, sy;
};

enum
{   // structure config property flags
    STRUCT_TRIANGULAR   = 0x01, // structure uses a triangular distribution
    STRUCT_CHUNK        = 0x02, // structure is checked for each chunk
    STRUCT_NETHER       = 0x10, // nether structure
    STRUCT_END          = 0x20, // end structure
};

enum
{
    LARGE_BIOMES            = 0x1,
    NO_BETA_OCEAN           = 0x2,
    FORCE_OCEAN_VARIANTS    = 0x4,
};

enum {
    SAMPLE_NO_SHIFT = 0x1,  // skip local distortions
    SAMPLE_NO_DEPTH = 0x2,  // skip depth sampling for vertical biomes
    SAMPLE_NO_BIOME = 0x4,  // do not apply climate noise to biome mapping
};

enum Dimension
{
    DIM_NETHER      =   -1,
    DIM_OVERWORLD   =    0,
    DIM_END         =   +1,
    DIM_UNDEF       = 1000,
};

enum StructureType
{
    Feature, // for locations of temple generation attempts pre 1.13
    Desert_Pyramid,
    Jungle_Temple, Jungle_Pyramid = Jungle_Temple,
    Swamp_Hut,
    Igloo,
    Village,
    Ocean_Ruin,
    Shipwreck,
    Monument,
    Mansion,
    Outpost,
    Ruined_Portal,
    Ruined_Portal_N,
    Ancient_City,
    Treasure,
    Mineshaft,
    Desert_Well,
    Geode,
    Fortress,
    Bastion,
    End_City,
    End_Gateway,
    End_Island,
    Trail_Ruins,
    Trial_Chambers,
    FEATURE_NUM
};

/* Minecraft versions */
enum MCVersion
{   // MC_1_X refers to the latest patch of the respective 1.X release.
    // NOTE: Development effort focuses on just the newest patch for each major
    // release. Minor releases and major versions <= 1.0 are experimental.
    MC_UNDEF,
    MC_B1_7,
    MC_B1_8,
    MC_1_0_0,  MC_1_0  = MC_1_0_0,
    MC_1_1_0,  MC_1_1  = MC_1_1_0,
    MC_1_2_5,  MC_1_2  = MC_1_2_5,
    MC_1_3_2,  MC_1_3  = MC_1_3_2,
    MC_1_4_7,  MC_1_4  = MC_1_4_7,
    MC_1_5_2,  MC_1_5  = MC_1_5_2,
    MC_1_6_4,  MC_1_6  = MC_1_6_4,
    MC_1_7_10, MC_1_7  = MC_1_7_10,
    MC_1_8_9,  MC_1_8  = MC_1_8_9,
    MC_1_9_4,  MC_1_9  = MC_1_9_4,
    MC_1_10_2, MC_1_10 = MC_1_10_2,
    MC_1_11_2, MC_1_11 = MC_1_11_2,
    MC_1_12_2, MC_1_12 = MC_1_12_2,
    MC_1_13_2, MC_1_13 = MC_1_13_2,
    MC_1_14_4, MC_1_14 = MC_1_14_4,
    MC_1_15_2, MC_1_15 = MC_1_15_2,
    MC_1_16_1,
    MC_1_16_5, MC_1_16 = MC_1_16_5,
    MC_1_17_1, MC_1_17 = MC_1_17_1,
    MC_1_18_2, MC_1_18 = MC_1_18_2,
    MC_1_19_2,
    MC_1_19,    // 1.19.3 - 1.19.4
    MC_1_20,
    MC_1_21,
    MC_NEWEST = MC_1_21,
};

enum BiomeID
{
    none = -1,
    // 0
    ocean = 0,
    plains,
    desert,
    mountains,                  extremeHills = mountains,
    forest,
    taiga,
    swamp,                      swampland = swamp,
    river,
    nether_wastes,              hell = nether_wastes,
    the_end,                    sky = the_end,
    // 10
    frozen_ocean,               frozenOcean = frozen_ocean,
    frozen_river,               frozenRiver = frozen_river,
    snowy_tundra,               icePlains = snowy_tundra,
    snowy_mountains,            iceMountains = snowy_mountains,
    mushroom_fields,            mushroomIsland = mushroom_fields,
    mushroom_field_shore,       mushroomIslandShore = mushroom_field_shore,
    beach,
    desert_hills,               desertHills = desert_hills,
    wooded_hills,               forestHills = wooded_hills,
    taiga_hills,                taigaHills = taiga_hills,
    // 20
    mountain_edge,              extremeHillsEdge = mountain_edge,
    jungle,
    jungle_hills,               jungleHills = jungle_hills,
    jungle_edge,                jungleEdge = jungle_edge,
    deep_ocean,                 deepOcean = deep_ocean,
    stone_shore,                stoneBeach = stone_shore,
    snowy_beach,                coldBeach = snowy_beach,
    birch_forest,               birchForest = birch_forest,
    birch_forest_hills,         birchForestHills = birch_forest_hills,
    dark_forest,                roofedForest = dark_forest,
    // 30
    snowy_taiga,                coldTaiga = snowy_taiga,
    snowy_taiga_hills,          coldTaigaHills = snowy_taiga_hills,
    giant_tree_taiga,           megaTaiga = giant_tree_taiga,
    giant_tree_taiga_hills,     megaTaigaHills = giant_tree_taiga_hills,
    wooded_mountains,           extremeHillsPlus = wooded_mountains,
    savanna,
    savanna_plateau,            savannaPlateau = savanna_plateau,
    badlands,                   mesa = badlands,
    wooded_badlands_plateau,    mesaPlateau_F = wooded_badlands_plateau,
    badlands_plateau,           mesaPlateau = badlands_plateau,
    // 40  --  1.13
    small_end_islands,
    end_midlands,
    end_highlands,
    end_barrens,
    warm_ocean,                 warmOcean = warm_ocean,
    lukewarm_ocean,             lukewarmOcean = lukewarm_ocean,
    cold_ocean,                 coldOcean = cold_ocean,
    deep_warm_ocean,            warmDeepOcean = deep_warm_ocean,
    deep_lukewarm_ocean,        lukewarmDeepOcean = deep_lukewarm_ocean,
    deep_cold_ocean,            coldDeepOcean = deep_cold_ocean,
    // 50
    deep_frozen_ocean,          frozenDeepOcean = deep_frozen_ocean,
    // Alpha 1.2 - Beta 1.7
    seasonal_forest,
    rainforest,
    shrubland,


    the_void = 127,

    // mutated variants
    sunflower_plains                = plains+128,
    desert_lakes                    = desert+128,
    gravelly_mountains              = mountains+128,
    flower_forest                   = forest+128,
    taiga_mountains                 = taiga+128,
    swamp_hills                     = swamp+128,
    ice_spikes                      = snowy_tundra+128,
    modified_jungle                 = jungle+128,
    modified_jungle_edge            = jungle_edge+128,
    tall_birch_forest               = birch_forest+128,
    tall_birch_hills                = birch_forest_hills+128,
    dark_forest_hills               = dark_forest+128,
    snowy_taiga_mountains           = snowy_taiga+128,
    giant_spruce_taiga              = giant_tree_taiga+128,
    giant_spruce_taiga_hills        = giant_tree_taiga_hills+128,
    modified_gravelly_mountains     = wooded_mountains+128,
    shattered_savanna               = savanna+128,
    shattered_savanna_plateau       = savanna_plateau+128,
    eroded_badlands                 = badlands+128,
    modified_wooded_badlands_plateau = wooded_badlands_plateau+128,
    modified_badlands_plateau       = badlands_plateau+128,
    // 1.14
    bamboo_jungle                   = 168,
    bamboo_jungle_hills             = 169,
    // 1.16
    soul_sand_valley                = 170,
    crimson_forest                  = 171,
    warped_forest                   = 172,
    basalt_deltas                   = 173,
    // 1.17
    dripstone_caves                 = 174,
    lush_caves                      = 175,
    // 1.18
    meadow                          = 177,
    grove                           = 178,
    snowy_slopes                    = 179,
    jagged_peaks                    = 180,
    frozen_peaks                    = 181,
    stony_peaks                     = 182,
    old_growth_birch_forest         = tall_birch_forest,
    old_growth_pine_taiga           = giant_tree_taiga,
    old_growth_spruce_taiga         = giant_spruce_taiga,
    snowy_plains                    = snowy_tundra,
    sparse_jungle                   = jungle_edge,
    stony_shore                     = stone_shore,
    windswept_hills                 = mountains,
    windswept_forest                = wooded_mountains,
    windswept_gravelly_hills        = gravelly_mountains,
    windswept_savanna               = shattered_savanna,
    wooded_badlands                 = wooded_badlands_plateau,
    // 1.19
    deep_dark                       = 183,
    mangrove_swamp                  = 184,
    // 1.20
    cherry_grove                    = 185,
};

#define LAYER_INIT_SHA          (~0ULL)


enum BiomeTempCategory
{
    Oceanic, Warm, Lush, Cold, Freezing, Special
};

/* Enumeration of the layer indices in the layer stack. */
enum LayerId
{
    // new                  [[deprecated]]
    L_CONTINENT_4096 = 0,   L_ISLAND_4096 = L_CONTINENT_4096,
    L_ZOOM_4096,                                                    // b1.8
    L_LAND_4096,                                                    // b1.8
    L_ZOOM_2048,
    L_LAND_2048,            L_ADD_ISLAND_2048 = L_LAND_2048,
    L_ZOOM_1024,
    L_LAND_1024_A,          L_ADD_ISLAND_1024A = L_LAND_1024_A,
    L_LAND_1024_B,          L_ADD_ISLAND_1024B = L_LAND_1024_B,     // 1.7+
    L_LAND_1024_C,          L_ADD_ISLAND_1024C = L_LAND_1024_C,     // 1.7+
    L_ISLAND_1024,          L_REMOVE_OCEAN_1024 = L_ISLAND_1024,    // 1.7+
    L_SNOW_1024,            L_ADD_SNOW_1024 = L_SNOW_1024,
    L_LAND_1024_D,          L_ADD_ISLAND_1024D = L_LAND_1024_D,     // 1.7+
    L_COOL_1024,            L_COOL_WARM_1024 = L_COOL_1024,         // 1.7+
    L_HEAT_1024,            L_HEAT_ICE_1024 = L_HEAT_1024,          // 1.7+
    L_SPECIAL_1024,                                                 // 1.7+
    L_ZOOM_512,
    L_LAND_512,                                                     // 1.6-
    L_ZOOM_256,
    L_LAND_256,             L_ADD_ISLAND_256 = L_LAND_256,
    L_MUSHROOM_256,         L_ADD_MUSHROOM_256 = L_MUSHROOM_256,
    L_DEEP_OCEAN_256,                                               // 1.7+
    L_BIOME_256,
    L_BAMBOO_256,           L14_BAMBOO_256 = L_BAMBOO_256,          // 1.14+
    L_ZOOM_128,
    L_ZOOM_64,
    L_BIOME_EDGE_64,
    L_NOISE_256,            L_RIVER_INIT_256 = L_NOISE_256,
    L_ZOOM_128_HILLS,
    L_ZOOM_64_HILLS,
    L_HILLS_64,
    L_SUNFLOWER_64,         L_RARE_BIOME_64 = L_SUNFLOWER_64,       // 1.7+
    L_ZOOM_32,
    L_LAND_32,              L_ADD_ISLAND_32 = L_LAND_32,
    L_ZOOM_16,
    L_SHORE_16,             // NOTE: in 1.0 this slot is scale 1:32
    L_SWAMP_RIVER_16,                                               // 1.6-
    L_ZOOM_8,
    L_ZOOM_4,
    L_SMOOTH_4,
    L_ZOOM_128_RIVER,
    L_ZOOM_64_RIVER,
    L_ZOOM_32_RIVER,
    L_ZOOM_16_RIVER,
    L_ZOOM_8_RIVER,
    L_ZOOM_4_RIVER,
    L_RIVER_4,
    L_SMOOTH_4_RIVER,
    L_RIVER_MIX_4,
    L_OCEAN_TEMP_256,       L13_OCEAN_TEMP_256 = L_OCEAN_TEMP_256,  // 1.13+
    L_ZOOM_128_OCEAN,       L13_ZOOM_128 = L_ZOOM_128_OCEAN,        // 1.13+
    L_ZOOM_64_OCEAN,        L13_ZOOM_64 = L_ZOOM_64_OCEAN,          // 1.13+
    L_ZOOM_32_OCEAN,        L13_ZOOM_32 = L_ZOOM_32_OCEAN,          // 1.13+
    L_ZOOM_16_OCEAN,        L13_ZOOM_16 = L_ZOOM_16_OCEAN,          // 1.13+
    L_ZOOM_8_OCEAN,         L13_ZOOM_8 = L_ZOOM_8_OCEAN,            // 1.13+
    L_ZOOM_4_OCEAN,         L13_ZOOM_4 = L_ZOOM_4_OCEAN,            // 1.13+
    L_OCEAN_MIX_4,          L13_OCEAN_MIX_4 = L_OCEAN_MIX_4,        // 1.13+

    L_VORONOI_1,            L_VORONOI_ZOOM_1 = L_VORONOI_1,

    // largeBiomes layers
    L_ZOOM_LARGE_A,
    L_ZOOM_LARGE_B,
    L_ZOOM_L_RIVER_A,
    L_ZOOM_L_RIVER_B,

    L_NUM
};

struct Layer;
typedef int (mapfunc_t)(const struct Layer *, int *, int, int, int, int);

STRUCT(Pos)  { int x, z; };

STRUCT(Layer)
{
    mapfunc_t *getMap;

    int8_t mc;          // minecraft version
    int8_t zoom;        // zoom factor of layer
    int8_t edge;        // maximum border required from parent layer
    int scale;          // scale of this layer (cell = scale x scale blocks)

    uint64_t layerSalt; // processed salt or initialization mode
    uint64_t startSalt; // (depends on world seed) used to step PRNG forward
    uint64_t startSeed; // (depends on world seed) start for chunk seeds

    void *noise;        // (depends on world seed) noise map data
    void *data;         // generic data for custom layers

    Layer *p, *p2;      // parent layers
};

STRUCT(PerlinNoise)
{
    uint8_t d[256+1];
    uint8_t h2;
    double a, b, c;
    double amplitude;
    double lacunarity;
    double d2;
    double t2;
};

STRUCT(OctaveNoise)
{
    int octcnt;
    PerlinNoise *octaves;
};

STRUCT(DoublePerlinNoise)
{
    double amplitude;
    OctaveNoise octA;
    OctaveNoise octB;
};

// Overworld biome generator up to 1.17
STRUCT(LayerStack)
{
    Layer layers[L_NUM];
    Layer *entry_1;     // entry scale (1:1) [L_VORONOI_1]
    Layer *entry_4;     // entry scale (1:4) [L_RIVER_MIX_4|L_OCEAN_MIX_4]
    // unofficial entries for other scales (latest sensible layers):
    Layer *entry_16;    // [L_SWAMP_RIVER_16|L_SHORE_16]
    Layer *entry_64;    // [L_HILLS_64|L_SUNFLOWER_64]
    Layer *entry_256;   // [L_BIOME_256|L_BAMBOO_256]
    PerlinNoise oceanRnd;
};

STRUCT(SurfaceNoise)
{
    double xzScale, yScale;
    double xzFactor, yFactor;
    OctaveNoise octmin;
    OctaveNoise octmax;
    OctaveNoise octmain;
    OctaveNoise octsurf;
    OctaveNoise octdepth;
    PerlinNoise oct[16+16+8+4+16];
};

STRUCT(SurfaceNoiseBeta)
{
    OctaveNoise octmin;
    OctaveNoise octmax;
    OctaveNoise octmain;
    OctaveNoise octcontA;
    OctaveNoise octcontB;
    PerlinNoise oct[16+16+8+10+16];
};

STRUCT(SeaLevelColumnNoiseBeta)
{
    double contASample;
    double contBSample;
    double minSample[2];
    double maxSample[2];
    double mainSample[2];
};

STRUCT(Spline)
{
    int len, typ;
    float loc[12];
    float der[12];
    Spline *val[12];
};

STRUCT(FixSpline)
{
    int len;
    float val;
};

STRUCT(SplineStack)
{   // the stack size here is just sufficient for overworld generation
    Spline stack[42];
    FixSpline fstack[151];
    int len, flen;
};


enum
{
    NP_TEMPERATURE      = 0,
    NP_HUMIDITY         = 1,
    NP_CONTINENTALNESS  = 2,
    NP_EROSION          = 3,
    NP_SHIFT            = 4, NP_DEPTH = NP_SHIFT, // not a real climate
    NP_WEIRDNESS        = 5,
    NP_MAX
};
// Overworld biome generator for 1.18+
STRUCT(BiomeNoise)
{
    DoublePerlinNoise climate[NP_MAX];
    PerlinNoise oct[2*23]; // buffer for octaves in double perlin noise
    Spline *sp;
    SplineStack ss;
    int nptype;
    int mc;
};
// Overworld biome generator for pre-Beta 1.8
STRUCT(BiomeNoiseBeta)
{
    OctaveNoise climate[3];
    PerlinNoise oct[10];
    int nptype;
    int mc;
};


STRUCT(BiomeTree)
{
    const uint32_t *steps;
    const int32_t  *param;
    const uint64_t *nodes;
    uint32_t order;
    uint32_t len;
};

STRUCT(Generator)
{
    int mc;
    int dim;
    uint32_t flags;
    uint64_t seed;
    uint64_t sha;

    union {
        struct { // MC 1.0 - 1.17
            LayerStack ls;
            Layer xlayer[5]; // buffer for custom entry layers @{1,4,16,64,256}
            Layer *entry;
        };
        struct { // MC 1.18
            BiomeNoise bn;
        };
        struct { // MC A1.2 - B1.7
            BiomeNoiseBeta bnb;
            //SurfaceNoiseBeta snb;
        };
    };
};

enum { SP_CONTINENTALNESS, SP_EROSION, SP_RIDGES, SP_WEIRDNESS };

__device__ __host__ static void addSplineVal(Spline *rsp, float loc, Spline *val, float der)
{
    rsp->loc[rsp->len] = loc;
    rsp->val[rsp->len] = val;
    rsp->der[rsp->len] = der;
    rsp->len++;
    //if (rsp->len > 12) {
    //    printf("addSplineVal(): too many spline points\n");
    //    exit(1);
    //}
}

__device__ __host__ static Spline *createFixSpline(SplineStack *ss, float val)
{
    FixSpline *sp = &ss->fstack[ss->flen++];
    sp->len = 1;
    sp->val = val;
    return (Spline*)sp;
}

__device__ __host__ static float getOffsetValue(float weirdness, float continentalness)
{
    float f0 = 1.0F - (1.0F - continentalness) * 0.5F;
    float f1 = 0.5F * (1.0F - continentalness);
    float f2 = (weirdness + 1.17F) * 0.46082947F;
    float off = f2 * f0 - f1;
    if (weirdness < -0.7F)
        return off > -0.2222F ? off : -0.2222F;
    else
        return off > 0 ? off : 0;
}

__device__ __host__ static Spline *createSpline_38219(SplineStack *ss, float f, int bl)
{
    Spline *sp = &ss->stack[ss->len++];
    sp->typ = SP_RIDGES;

    float i = getOffsetValue(-1.0F, f);
    float k = getOffsetValue( 1.0F, f);
    float l = 1.0F - (1.0F - f) * 0.5F;
    float u = 0.5F * (1.0F - f);
    l = u / (0.46082947F * l) - 1.17F;

    if (-0.65F < l && l < 1.0F)
    {
        float p, q, r, s;
        u = getOffsetValue(-0.65F, f);
        p = getOffsetValue(-0.75F, f);
        q = (p - i) * 4.0F;
        r = getOffsetValue(l, f);
        s = (k - r) / (1.0F - l);

        addSplineVal(sp, -1.0F,     createFixSpline(ss, i), q);
        addSplineVal(sp, -0.75F,    createFixSpline(ss, p), 0);
        addSplineVal(sp, -0.65F,    createFixSpline(ss, u), 0);
        addSplineVal(sp, l-0.01F,   createFixSpline(ss, r), 0);
        addSplineVal(sp, l,         createFixSpline(ss, r), s);
        addSplineVal(sp, 1.0F,      createFixSpline(ss, k), s);
    }
    else
    {
        u = (k - i) * 0.5F;
        if (bl) {
            addSplineVal(sp, -1.0F, createFixSpline(ss, i > 0.2 ? i : 0.2), 0);
            addSplineVal(sp,  0.0F, createFixSpline(ss, lerp(0.5F, i, k)), u);
        } else {
            addSplineVal(sp, -1.0F, createFixSpline(ss, i), u);
        }
        addSplineVal(sp, 1.0F,      createFixSpline(ss, k), u);
    }
    return sp;
}

__device__ __host__ static Spline *createFlatOffsetSpline(
    SplineStack *ss, float f, float g, float h, float i, float j, float k)
{
    Spline *sp = &ss->stack[ss->len++];
    sp->typ = SP_RIDGES;

    float l = 0.5F * (g - f); if (l < k) l = k;
    float m = 5.0F * (h - g);

    addSplineVal(sp, -1.0F, createFixSpline(ss, f), l);
    addSplineVal(sp, -0.4F, createFixSpline(ss, g), l < m ? l : m);
    addSplineVal(sp,  0.0F, createFixSpline(ss, h), m);
    addSplineVal(sp,  0.4F, createFixSpline(ss, i), 2.0F*(i-h));
    addSplineVal(sp,  1.0F, createFixSpline(ss, j), 0.7F*(j-i));

    return sp;
}

__device__ __host__ static Spline *createLandSpline(
    SplineStack *ss, float f, float g, float h, float i, float j, float k, int bl)
{
    Spline *sp1 = createSpline_38219(ss, lerp(i, 0.6F, 1.5F), bl);
    Spline *sp2 = createSpline_38219(ss, lerp(i, 0.6F, 1.0F), bl);
    Spline *sp3 = createSpline_38219(ss, i, bl);
    const float ih = 0.5F * i;
    Spline *sp4 = createFlatOffsetSpline(ss, f-0.15F, ih, ih, ih, i*0.6F, 0.5F);
    Spline *sp5 = createFlatOffsetSpline(ss, f, j*i, g*i, ih, i*0.6F, 0.5F);
    Spline *sp6 = createFlatOffsetSpline(ss, f, j, j, g, h, 0.5F);
    Spline *sp7 = createFlatOffsetSpline(ss, f, j, j, g, h, 0.5F);

    Spline *sp8 = &ss->stack[ss->len++];
    sp8->typ = SP_RIDGES;
    addSplineVal(sp8, -1.0F, createFixSpline(ss, f), 0.0F);
    addSplineVal(sp8, -0.4F, sp6, 0.0F);
    addSplineVal(sp8,  0.0F, createFixSpline(ss, h + 0.07F), 0.0F);

    Spline *sp9 = createFlatOffsetSpline(ss, -0.02F, k, k, g, h, 0.0F);
    Spline *sp = &ss->stack[ss->len++];
    sp->typ = SP_EROSION;
    addSplineVal(sp, -0.85F, sp1, 0.0F);
    addSplineVal(sp, -0.7F,  sp2, 0.0F);
    addSplineVal(sp, -0.4F,  sp3, 0.0F);
    addSplineVal(sp, -0.35F, sp4, 0.0F);
    addSplineVal(sp, -0.1F,  sp5, 0.0F);
    addSplineVal(sp,  0.2F,  sp6, 0.0F);
    if (bl) {
        addSplineVal(sp, 0.4F,  sp7, 0.0F);
        addSplineVal(sp, 0.45F, sp8, 0.0F);
        addSplineVal(sp, 0.55F, sp8, 0.0F);
        addSplineVal(sp, 0.58F, sp7, 0.0F);
    }
    addSplineVal(sp, 0.7F, sp9, 0.0F);
    return sp;
}
__device__ __host__ float getSpline(const Spline *sp, const float *vals) {
    if (!sp || sp->len <= 0 || sp->len >= 12)
    {
        printf("getSpline(): bad parameters\n");
        // exit(1);
    }

    if (sp->len == 1)
        return ((FixSpline*)sp)->val;

    float f = vals[sp->typ];
    int i;

    for (i = 0; i < sp->len; i++)
        if (sp->loc[i] >= f)
            break;
    if (i == 0 || i == sp->len) {
        if (i) i--;
        float v = getSpline(sp->val[i], vals);
        return v + sp->der[i] * (f - sp->loc[i]);
    }
    const Spline *sp1 = sp->val[i-1];
    const Spline *sp2 = sp->val[i];
    float g = sp->loc[i-1];
    float h = sp->loc[i];
    float k = (f - g) / (h - g);
    float l = sp->der[i-1];
    float m = sp->der[i];
    float n = getSpline(sp1, vals);
    float o = getSpline(sp2, vals);
    float p = l * (h - g) - (o - n);
    float q = -m * (h - g) + (o - n);
    float r = lerp(k, n, o) + k * (1.0F - k) * lerp(k, p, q);
    return r;
}

__device__ __host__ ATTR(hot, const)
static inline double indexedLerp(uint8_t idx, double a, double b, double c)
{
   switch (idx & 0xf)
   {
   case 0:  return  a + b;
   case 1:  return -a + b;
   case 2:  return  a - b;
   case 3:  return -a - b;
   case 4:  return  a + c;
   case 5:  return -a + c;
   case 6:  return  a - c;
   case 7:  return -a - c;
   case 8:  return  b + c;
   case 9:  return -b + c;
   case 10: return  b - c;
   case 11: return -b - c;
   case 12: return  a + b;
   case 13: return -b + c;
   case 14: return -a + b;
   case 15: return -b - c;
   }
   UNREACHABLE();
   return 0;
}

__device__ __host__ double samplePerlin(const PerlinNoise *noise, double d1, double d2, double d3,
        double yamp, double ymin)
{
    uint8_t h1, h2, h3;
    double t1, t2, t3;

    if (d2 == 0.0)
    {
        d2 = noise->d2;
        h2 = noise->h2;
        t2 = noise->t2;
    }
    else
    {
        d2 += noise->b;
        double i2 = floor(d2);
        d2 -= i2;
        h2 = (int) i2;
        t2 = d2*d2*d2 * (d2 * (d2*6.0-15.0) + 10.0);
    }

    d1 += noise->a;
    d3 += noise->c;

    double i1 = floor(d1);
    double i3 = floor(d3);
    d1 -= i1;
    d3 -= i3;

    h1 = (int) i1;
    h3 = (int) i3;

    t1 = d1*d1*d1 * (d1 * (d1*6.0-15.0) + 10.0);
    t3 = d3*d3*d3 * (d3 * (d3*6.0-15.0) + 10.0);

    if (yamp)
    {
        double yclamp = ymin < d2 ? ymin : d2;
        d2 -= floor(yclamp / yamp) * yamp;
    }

    const uint8_t *idx = noise->d;

#if 1
    // try to promote optimizations that can utilize the {xh, xl} registers
    typedef struct vec2 { uint8_t a, b; } vec2;

    vec2 v1 = { idx[h1], idx[h1+1] };
    v1.a += h2;
    v1.b += h2;

    vec2 v2 = { idx[v1.a], idx[v1.a+1] };
    vec2 v3 = { idx[v1.b], idx[v1.b+1] };
    v2.a += h3;
    v2.b += h3;
    v3.a += h3;
    v3.b += h3;

    vec2 v4 = { idx[v2.a], idx[v2.a+1] };
    vec2 v5 = { idx[v2.b], idx[v2.b+1] };
    vec2 v6 = { idx[v3.a], idx[v3.a+1] };
    vec2 v7 = { idx[v3.b], idx[v3.b+1] };

    double l1 = indexedLerp(v4.a, d1,   d2,   d3);
    double l5 = indexedLerp(v4.b, d1,   d2,   d3-1);
    double l2 = indexedLerp(v6.a, d1-1, d2,   d3);
    double l6 = indexedLerp(v6.b, d1-1, d2,   d3-1);
    double l3 = indexedLerp(v5.a, d1,   d2-1, d3);
    double l7 = indexedLerp(v5.b, d1,   d2-1, d3-1);
    double l4 = indexedLerp(v7.a, d1-1, d2-1, d3);
    double l8 = indexedLerp(v7.b, d1-1, d2-1, d3-1);
#else
    uint8_t a1 = idx[h1]   + h2;
    uint8_t b1 = idx[h1+1] + h2;

    uint8_t a2 = idx[a1]   + h3;
    uint8_t b2 = idx[b1]   + h3;
    uint8_t a3 = idx[a1+1] + h3;
    uint8_t b3 = idx[b1+1] + h3;

    double l1 = indexedLerp(idx[a2],   d1,   d2,   d3);
    double l2 = indexedLerp(idx[b2],   d1-1, d2,   d3);
    double l3 = indexedLerp(idx[a3],   d1,   d2-1, d3);
    double l4 = indexedLerp(idx[b3],   d1-1, d2-1, d3);
    double l5 = indexedLerp(idx[a2+1], d1,   d2,   d3-1);
    double l6 = indexedLerp(idx[b2+1], d1-1, d2,   d3-1);
    double l7 = indexedLerp(idx[a3+1], d1,   d2-1, d3-1);
    double l8 = indexedLerp(idx[b3+1], d1-1, d2-1, d3-1);
#endif

    l1 = lerp(t1, l1, l2);
    l3 = lerp(t1, l3, l4);
    l5 = lerp(t1, l5, l6);
    l7 = lerp(t1, l7, l8);

    l1 = lerp(t2, l1, l3);
    l5 = lerp(t2, l5, l7);

    return lerp(t3, l1, l5);
}

__device__ __host__ double maintainPrecision(double x)
{   // This is a highly performance critical function that is used to correct
    // progressing errors from float-maths. However, since cubiomes uses
    // doubles anyway, this seems useless in practice.

    //return x - round(x / 33554432.0) * 33554432.0;
    return x;
}

__device__ __host__ double sampleOctave(const OctaveNoise *noise, double x, double y, double z)
{
    double v = 0;
    int i;
    for (i = 0; i < noise->octcnt; i++)
    {
        PerlinNoise *p = noise->octaves + i;
        double lf = p->lacunarity;
        double ax = maintainPrecision(x * lf);
        double ay = maintainPrecision(y * lf);
        double az = maintainPrecision(z * lf);
        double pv = samplePerlin(p, ax, ay, az, 0, 0);
        v += p->amplitude * pv;
    }
    return v;
}

__device__ __host__ double sampleDoublePerlin(const DoublePerlinNoise *noise,
        double x, double y, double z)
{
    const double f = 337.0 / 331.0;
    double v = 0;

    v += sampleOctave(&noise->octA, x, y, z);
    v += sampleOctave(&noise->octB, x*f, y*f, z*f);

    return v * noise->amplitude;
}

__device__ __host__ void initBiomeNoise(BiomeNoise *bn, int mc)
{
    SplineStack *ss = &bn->ss;
    memset(ss, 0, sizeof(*ss));
    Spline *sp = &ss->stack[ss->len++];
    sp->typ = SP_CONTINENTALNESS;

    Spline *sp1 = createLandSpline(ss, -0.15F, 0.00F, 0.0F, 0.1F, 0.00F, -0.03F, 0);
    Spline *sp2 = createLandSpline(ss, -0.10F, 0.03F, 0.1F, 0.1F, 0.01F, -0.03F, 0);
    Spline *sp3 = createLandSpline(ss, -0.10F, 0.03F, 0.1F, 0.7F, 0.01F, -0.03F, 1);
    Spline *sp4 = createLandSpline(ss, -0.05F, 0.03F, 0.1F, 1.0F, 0.01F,  0.01F, 1);

    addSplineVal(sp, -1.10F, createFixSpline(ss,  0.044F), 0.0F);
    addSplineVal(sp, -1.02F, createFixSpline(ss, -0.2222F), 0.0F);
    addSplineVal(sp, -0.51F, createFixSpline(ss, -0.2222F), 0.0F);
    addSplineVal(sp, -0.44F, createFixSpline(ss, -0.12F), 0.0F);
    addSplineVal(sp, -0.18F, createFixSpline(ss, -0.12F), 0.0F);
    addSplineVal(sp, -0.16F, sp1, 0.0F);
    addSplineVal(sp, -0.15F, sp1, 0.0F);
    addSplineVal(sp, -0.10F, sp2, 0.0F);
    addSplineVal(sp,  0.25F, sp3, 0.0F);
    addSplineVal(sp,  1.00F, sp4, 0.0F);

    bn->sp = sp;
    bn->mc = mc;
}

__device__ __host__ double sampleClimatePara(const BiomeNoise *bn, int64_t *np, double x, double z)
{
    if (bn->nptype == NP_DEPTH)
    {
        float c, e, w;
        c = sampleDoublePerlin(bn->climate + NP_CONTINENTALNESS, x, 0, z);
        e = sampleDoublePerlin(bn->climate + NP_EROSION, x, 0, z);
        w = sampleDoublePerlin(bn->climate + NP_WEIRDNESS, x, 0, z);

        float np_param[] = {
            c, e, -3.0F * ( fabsf( fabsf(w) - 0.6666667F ) - 0.33333334F ), w,
        };
        printf("GETTING SPLINE\n");
        double off = getSpline(bn->sp, np_param) + 0.015F;
        int y = 0;
        float d = 1.0 - (y * 4) / 128.0 - 83.0/160.0 + off;
        if (np)
        {
            np[2] = (int64_t)(10000.0F*c);
            np[3] = (int64_t)(10000.0F*e);
            np[4] = (int64_t)(10000.0F*d);
            np[5] = (int64_t)(10000.0F*w);
        }
        return d;
    }
    double p = sampleDoublePerlin(bn->climate + bn->nptype, x, 0, z);
    if (np)
        np[bn->nptype] = (int64_t)(10000.0F*p);
    return p;
}

__device__ __host__ uint64_t get_np_dist(const uint64_t np[6], const BiomeTree *bt, int idx)
{
    uint64_t ds = 0, node = bt->nodes[idx];
    uint64_t a, b, d;
    uint32_t i;

    for (i = 0; i < 6; i++)
    {
        idx = (node >> 8*i) & 0xFF;
        a = np[i] - bt->param[2*idx + 1];
        b = bt->param[2*idx + 0] - np[i];
        d = (int64_t)a > 0 ? a : (int64_t)b > 0 ? b : 0;
        d = d * d;
        ds += d;
    }
    return ds;
}

__device__ __host__ int get_resulting_node_new(const uint64_t np[6], const BiomeTree *bt) {
    uint64_t best_distance = 9999999999;
    int best_biome = -1;
    for (size_t i = 0; i < 9112; i++) {
        uint64_t node = bt->nodes[i];
        int is_child = (node >> 8 * 7) & 0xFF;

        if (is_child != 0xFF) {
            continue;
        }

        int biome = (node >> 8 * 6) & 0xFF;

        // best_distance = biome;
        uint64_t dist = get_np_dist(np, bt, i);
        // (void)dist;
        if (dist < best_distance) {
            best_distance = dist;
            best_biome = biome;
        }

        // Highly questionable speedup
        if (best_distance < 5) {
            break;
        }
    }
    return best_biome;
}

#include <inttypes.h>

enum { btree20_order = 6 };

__device__ static const uint32_t btree20_steps[] = { 1555, 259, 43, 7, 1, 0 };

__device__ static const int32_t btree20_param[][2] =
{
    {-12000,-10500},{-12000, -4550},{-12000, 10000},{-10500, -4550}, // 00-03
    {-10500, -1900},{-10500, 10000},{-10000, -9333},{-10000, -7799}, // 04-07
    {-10000, -7666},{-10000, -5666},{-10000, -4500},{-10000, -4000}, // 08-0B
    {-10000, -3750},{-10000, -3500},{-10000, -2666},{-10000, -2225}, // 0C-0F
    {-10000, -1500},{-10000, -1000},{-10000,  -500},{-10000,   500}, // 10-13
    {-10000,  1000},{-10000,  2000},{-10000,  2666},{-10000,  3000}, // 14-17
    {-10000,  4000},{-10000,  5500},{-10000,  5666},{-10000,  9333}, // 18-1B
    {-10000, 10000},{ -9333, -7666},{ -9333, -5666},{ -9333, -4000}, // 1C-1F
    { -9333, -2666},{ -9333,  2666},{ -9333,  4000},{ -9333,  5666}, // 20-23
    { -9333, 10000},{ -7799, -3750},{ -7799, -2225},{ -7666, -5666}, // 24-27
    { -7666, -4000},{ -7666, -2666},{ -7666,  -500},{ -7666,   500}, // 28-2B
    { -7666,  4000},{ -7666,  5666},{ -5666, -4000},{ -5666, -2666}, // 2C-2F
    { -5666,  -500},{ -5666,   500},{ -5666,  2666},{ -5666,  4000}, // 30-33
    { -5666,  5666},{ -5666,  7666},{ -4550, -1900},{ -4500, -1500}, // 34-37
    { -4500,  2000},{ -4500,  5500},{ -4500, 10000},{ -4000, -2666}, // 38-3B
    { -4000,  -500},{ -4000,  2666},{ -4000,  4000},{ -4000,  5666}, // 3C-3F
    { -4000,  9333},{ -4000, 10000},{ -3750, -2225},{ -3750,   500}, // 40-43
    { -3750,  5500},{ -3500, -1000},{ -3500,  1000},{ -3500,  3000}, // 44-47
    { -3500, 10000},{ -2666,  -500},{ -2666,   500},{ -2666,  2666}, // 48-4B
    { -2666,  4000},{ -2666,  5666},{ -2666,  9333},{ -2666, 10000}, // 4C-4F
    { -2225,   500},{ -2225,  4500},{ -1900, -1100},{ -1900,   300}, // 50-53
    { -1900,  3000},{ -1900, 10000},{ -1500,  2000},{ -1500,  5500}, // 54-57
    { -1500, 10000},{ -1100,   300},{ -1100,  3000},{ -1100, 10000}, // 58-5B
    { -1000,  1000},{ -1000,  3000},{ -1000, 10000},{  -500,   500}, // 5C-5F
    {  -500,  2666},{     0,     0},{     0, 10000},{     0, 11000}, // 60-63
    {   300,  3000},{   300, 10000},{   500,  2666},{   500,  4000}, // 64-67
    {   500,  4500},{   500,  5500},{   500,  5666},{   500,  7666}, // 68-6B
    {   500,  9333},{   500, 10000},{  1000,  3000},{  1000, 10000}, // 6C-6F
    {  2000,  5500},{  2000,  9000},{  2000, 10000},{  2666,  4000}, // 70-73
    {  2666,  5666},{  2666,  7666},{  2666,  9333},{  2666, 10000}, // 74-77
    {  3000, 10000},{  4000,  5666},{  4000,  7666},{  4000,  9333}, // 78-7B
    {  4000, 10000},{  4500,  5500},{  4500, 10000},{  5500, 10000}, // 7C-7F
    {  5666,  7666},{  5666,  9333},{  5666, 10000},{  7000, 10000}, // 80-83
    {  7666,  9333},{  7666, 10000},{  8000, 10000},{  9333, 10000}, // 84-87
    { 10000, 10000},{ 10000, 11000},{ 11000, 11000},
};

__device__ static const uint64_t btree20_nodes[] =
{
    // Binary encoded biome parameter search tree for 1.20 (20w07a).
    //
    //   +-------------- If the top byte equals 0xFF, the node is a leaf and the
    //   |               second byte is the biome id, otherwise the two bytes
    //   |               are a short index to the first child node.
    //   |
    //   | +------------ Biome parameter index for 5 (weirdness)
    //   | | +---------- Biome parameter index for 4 (depth)
    //   | | | +-------- Biome parameter index for 3 (erosion)
    //   | | | | +------ Biome parameter index for 2 (continentalness)
    //   | | | | | +---- Biome parameter index for 1 (humidity)
    //   | | | | | | +-- Biome parameter index for 0 (temperature)
    //   | | | | | | |
    //   v v v v v v v
    0x0001616161616161,
    0x00021C621C021C1C,0x00031C621C021C1C,0x00041C621C051C1C,0x0005496251521456,
    0xFF10496151525C56,0xFF10496151524556,0xFF10496151520D56,0xFF10498851525C56,
    0xFF10498851524556,0xFF10498851520D56,0x000C496251521C57,0xFF10496151526E56,
    0xFF10496151527856,0xFF10496151520D70,0xFF10498851526E56,0xFF10498851527856,
    0xFF10498851520D70,0x0013496251524770,0xFF10496151525C70,0xFF10496151526E70,
    0xFF10496151524570,0xFF10498851525C70,0xFF10498851526E70,0xFF10498851524570,
    0x001A1C621C051C10,0xFF0A1C611C361C0A,0xFF0B5F6144551C0A,0xFF321C611C031C0A,
    0xFF0A1C881C361C0A,0xFF10498851527837,0xFF321C881C031C0A,0x0021496251521C72,
    0xFF0249615152457F,0xFF10496151527870,0xFF02496151520D7F,0xFF0249885152457F,
    0xFF10498851527870,0xFF02498851520D7F,0x0028496251525E7F,0xFF02496151525C7F,
    0xFF02496151526E7F,0xFF0249615152787F,0xFF02498851525C7F,0xFF02498851526E7F,
    0xFF0249885152787F,0x002F1C621C021C1C,0x00301C621C1C1C1C,0xFF075F6144551C3A,
    0xFFAF1C711C1C831C,0xFFAE1C711C861C1C,0xFF075F8844551C3A,0xFF0B5F8844551C0A,
    0xFF10668851520D37,0x0037666151521C10,0xFF1A666151525C0A,0xFF1A666151526E0A,
    0xFF1A66615152450A,0xFF10666151520D37,0xFF1A66615152780A,0xFF1A666151520D0A,
    0x003E666251521C10,0xFF10666151524537,0xFF1A668851525C0A,0xFF1A668851526E0A,
    0xFF1A66885152450A,0xFF1A66885152780A,0xFF1A668851520D0A,0x00451C621C361C3A,
    0xFF2D1C611C361C70,0xFF2C1C611C361C7F,0xFF001C881C361C56,0xFF2E1C881C361C37,
    0xFF2D1C881C361C70,0xFF2C1C881C361C7F,0x004C1C621C041C3A,0xFF001C611C361C56,
    0xFF2E1C611C361C37,0xFF181C881C031C56,0xFF311C881C031C37,0xFF301C881C031C70,
    0xFF2C1C881C031C7F,0x00531C621C011C1C,0xFF181C611C031C56,0xFF311C611C031C37,
    0xFF0E1C611C001C1C,0xFF301C611C031C70,0xFF2C1C611C031C7F,0xFF0E1C881C001C1C,
    0x005A666251521C3A,0x005B666151521C38,0xFF10666151525C56,0xFF10666151524556,
    0xFF10666151525C37,0xFF10666151526E37,0xFF10666151520D56,0xFF10666151527837,
    0x0062666151521C57,0xFF10666151526E56,0xFF10666151525C70,0xFF10666151526E70,
    0xFF10666151524570,0xFF10666151527856,0xFF10666151520D70,0x0069666151521C72,
    0xFF02666151525C7F,0xFF02666151526E7F,0xFF0266615152457F,0xFF10666151527870,
    0xFF0266615152787F,0xFF02666151520D7F,0x0070668851521C38,0xFF10668851524556,
    0xFF10668851525C37,0xFF10668851526E37,0xFF10668851524537,0xFF10668851520D56,
    0xFF10668851527837,0x0077668851521C57,0xFF10668851525C56,0xFF10668851526E56,
    0xFF10668851525C70,0xFF10668851524570,0xFF10668851527856,0xFF10668851520D70,
    0x007E668851521C72,0xFF10668851526E70,0xFF02668851525C7F,0xFF02668851526E7F,
    0xFF0266885152457F,0xFF10668851527870,0xFF02668851520D7F,0x0085126251551C1C,
    0x0086496251524837,0xFF10496151525C37,0xFF10496151526E37,0xFF10496151527837,
    0xFF10498851525C37,0xFF10498851526E37,0xFF10498851524537,0x008D496251521C10,
    0xFF10496151524537,0xFF10496151520D37,0xFF1A49615152780A,0xFF1A498851526E0A,
    0xFF10498851520D37,0xFF1A49885152780A,0x009449625152170A,0xFF1A496151525C0A,
    0xFF1A496151526E0A,0xFF1A49615152450A,0xFF1A498851525C0A,0xFF1A49885152450A,
    0xFF1A498851520D0A,0x009B126151551C1C,0xFF1A496151520D0A,0xFF0C1D6168555C0A,
    0xFF0C1D616855450A,0xFF0206616852787F,0xFF0C1D6168550D0A,0xFF020661685B787F,
    0x00A2066168551C72,0xFF10066168526E70,0xFF150661685B6E70,0xFF10066168527870,
    0xFF150661685B7870,0xFF02066168520D7F,0xFF020661685B0D7F,0x00A906616855477F,
    0xFF02066168525C7F,0xFF02066168526E7F,0xFF0206616852457F,0xFF020661685B5C7F,
    0xFF020661685B6E7F,0xFF020661685B457F,0x00B06D8851551C3A,0x00B1878850555C3A,
    0xFF04878850535C56,0xFF01878850535C70,0xFF04878850655C56,0xFF04878850655C37,
    0xFF02878850535C7F,0xFF01878850655C70,0x00B8878850555D3A,0xFF9B878850536E56,
    0xFF05878850536E37,0xFF17878850536E70,0xFF05878850656E37,0xFF02878850536E7F,
    0xFF25878850655C7F,0x00BF878850554558,0xFF01878850534556,0xFF23878850534570,
    0xFF01878850654556,0xFF0287885053457F,0xFF23878850654570,0xFFA587885065457F,
    0x00C66D8851556F3A,0xFF9B878850656E56,0xFF1D878850537856,0xFF0266885152787F,
    0xFF20878850537837,0xFF17878850656E70,0xFF26878850656E7F,0x00CD87885055783A,
    0xFFA8878850537870,0xFF1D878850657856,0xFF20878850657837,0xFF0287885053787F,
    0xFFA8878850657870,0xFF2687885065787F,0x00D4878850550D58,0xFF81878850530D56,
    0xFF23878850530D70,0xFF81878850650D56,0xFF02878850530D7F,0xFF23878850650D70,
    0xFFA5878850650D7F,0x00DB066168551C19,0x00DC066168555E56,0xFF10066168525C56,
    0xFF10066168526E56,0xFF040661685B5C56,0xFF1B0661685B6E56,0xFF10066168527856,
    0xFF1D0661685B7856,0x00E3066168551C38,0xFF10066168524556,0xFF010661685B4556,
    0xFF10066168520D56,0xFF10066168527837,0xFF840661685B0D56,0xFFA00661685B7837,
    0x00EA066168554737,0xFF10066168525C37,0xFF10066168526E37,0xFF10066168524537,
    0xFF040661685B5C37,0xFF050661685B6E37,0xFF010661685B4537,0x00F1066168551470,
    0xFF10066168525C70,0xFF10066168524570,0xFF040661685B5C70,0xFF230661685B4570,
    0xFF10066168520D70,0xFF230661685B0D70,0x00F8066168551C10,0xFF1A066168526E0A,
    0xFF10066168520D37,0xFF1E0661685B6E0A,0xFF010661685B0D37,0xFF1A06616852780A,
    0xFF050661685B780A,0x00FF06616855140A,0xFF1A066168525C0A,0xFF1A06616852450A,
    0xFF0C0661685B5C0A,0xFF0C0661685B450A,0xFF1A066168520D0A,0xFF0C0661685B0D0A,
    0x01061C6250551C1C,0x01071A6250551C19,0x0108068850551756,0xFF04068850535C56,
    0xFF1B068850536E56,0xFF01068850534556,0xFF04068850655C56,0xFF01068850654556,
    0xFF84068850650D56,0x010F1A6250551C38,0xFF20796150647837,0xFFB1796150786E37,
    0xFF20796150787837,0xFF84068850530D56,0xFFA0068850537837,0xFFA0068850657837,
    0x0116068850551C57,0xFF23068850534570,0xFF1B068850656E56,0xFF1D068850537856,
    0xFF23068850530D70,0xFF1D068850657856,0xFF23068850650D70,0x011D068850554737,
    0xFF04068850535C37,0xFF05068850536E37,0xFF01068850534537,0xFF04068850655C37,
    0xFF05068850656E37,0xFF01068850654537,0x0124068850551C10,0xFF1E068850536E0A,
    0xFF01068850530D37,0xFF1E068850656E0A,0xFF0506885053780A,0xFF01068850650D37,
    0xFF0506885065780A,0x012B06885055140A,0xFF0C068850535C0A,0xFF0C06885053450A,
    0xFF0C068850655C0A,0xFF0C06885065450A,0xFF0C068850530D0A,0xFF0C068850650D0A,
    0x01327A6150651C1C,0x0133796150651456,0xFF04796150645C56,0xFF01796150644556,
    0xFF04796150785C56,0xFF81796150640D56,0xFFB9796150784556,0xFFB9796150780D56,
    0x013A796150651C57,0xFF9B796150646E56,0xFF1D796150647856,0xFF1B796150786E56,
    0xFF23796150640D70,0xFF1D796150787856,0xFF24796150780D70,0x0141796150654770,
    0xFF01796150645C70,0xFF17796150646E70,0xFF23796150644570,0xFF04796150785C70,
    0xFF04796150786E70,0xFF24796150784570,0x0148796150651C72,0xFFA579615064457F,
    0xFFA8796150647870,0xFFA5796150640D7F,0xFFA579615078457F,0xFF15796150787870,
    0xFFA5796150780D7F,0x014F80615065140A,0xFF1E806150645C0A,0xFF0C80615064450A,
    0xFF0C806150785C0A,0xFF8C806150640D0A,0xFF0C80615078450A,0xFF8C806150780D0A,
    0x0156796150655E7F,0xFF25796150645C7F,0xFF26796150646E7F,0xFF25796150785C7F,
    0xFF2679615064787F,0xFF26796150786E7F,0xFF2679615078787F,0x015D806150651C1C,
    0x015E806150655E56,0xFF04806150645C56,0xFF9B806150646E56,0xFF04806150785C56,
    0xFF1D806150647856,0xFF1B806150786E56,0xFF1D806150787856,0x0165806150651C38,
    0xFF01806150644556,0xFF81806150640D56,0xFFB9806150784556,0xFF20806150647837,
    0xFFB9806150780D56,0xFF20806150787837,0x016C806150654737,0xFF04806150645C37,
    0xFF05806150646E37,0xFF01806150644537,0xFFB1806150785C37,0xFFB1806150786E37,
    0xFFB1806150784537,0x0173806150651470,0xFF01806150645C70,0xFF23806150644570,
    0xFF04806150785C70,0xFF23806150640D70,0xFF24806150784570,0xFF24806150780D70,
    0x017A806150651C10,0xFF1E806150646E0A,0xFF01806150640D37,0xFF0580615064780A,
    0xFF1E806150786E0A,0xFFB9806150780D37,0xFF1E80615078780A,0x0181806150651C72,
    0xFF17806150646E70,0xFFA8806150647870,0xFF04806150786E70,0xFFA5806150640D7F,
    0xFF15806150787870,0xFFA5806150780D7F,0x0188816150651C1C,0x0189846150651C38,
    0xFF01846150644556,0xFF81846150640D56,0xFFB9846150784556,0xFF20846150647837,
    0xFFB9846150780D56,0xFF20846150787837,0x0190846150654737,0xFF04846150645C37,
    0xFF05846150646E37,0xFF01846150644537,0xFFB1846150785C37,0xFFB1846150786E37,
    0xFFB1846150784537,0x0197816150654758,0xFF04846150645C56,0xFF9B846150646E56,
    0xFF04846150785C56,0xFF1B846150786E56,0xFFA580615064457F,0xFFA580615078457F,
    0x019E846150651C10,0xFF1E846150646E0A,0xFF01846150640D37,0xFF0584615064780A,
    0xFF1E846150786E0A,0xFFB9846150780D37,0xFF1E84615078780A,0x01A584615065140A,
    0xFF1E846150645C0A,0xFF0C84615064450A,0xFF0C846150785C0A,0xFF8C846150640D0A,
    0xFF0C84615078450A,0xFF8C846150780D0A,0x01AC806150655E7F,0xFF25806150645C7F,
    0xFF26806150646E7F,0xFF25806150785C7F,0xFF2680615064787F,0xFF26806150786E7F,
    0xFF2680615078787F,0x01B3856150551C1C,0x01B4856150551C39,0xFF01876150534537,
    0xFF1D846150647856,0xFF01876150654537,0xFF23846150640D70,0xFF1D846150787856,
    0xFF24846150780D70,0x01BB846150654770,0xFF01846150645C70,0xFF17846150646E70,
    0xFF23846150644570,0xFF04846150785C70,0xFF04846150786E70,0xFF24846150784570,
    0x01C2876150551C10,0xFF1E876150536E0A,0xFF01876150530D37,0xFF1E876150656E0A,
    0xFF0587615053780A,0xFF01876150650D37,0xFF0587615065780A,0x01C9846150651C72,
    0xFFA584615064457F,0xFFA8846150647870,0xFFA5846150640D7F,0xFFA584615078457F,
    0xFF15846150787870,0xFFA5846150780D7F,0x01D087615055140A,0xFF1E876150535C0A,
    0xFF0C87615053450A,0xFF1E876150655C0A,0xFF0C87615065450A,0xFF8C876150530D0A,
    0xFF8C876150650D0A,0x01D7846150655E7F,0xFF25846150645C7F,0xFF26846150646E7F,
    0xFF25846150785C7F,0xFF2684615064787F,0xFF26846150786E7F,0xFF2684615078787F,
    0x01DE876150551C3A,0x01DF876150551456,0xFF04876150535C56,0xFF01876150534556,
    0xFF04876150655C56,0xFF01876150654556,0xFF81876150530D56,0xFF81876150650D56,
    0x01E6876150551C57,0xFF9B876150536E56,0xFF9B876150656E56,0xFF1D876150537856,
    0xFF23876150530D70,0xFF1D876150657856,0xFF23876150650D70,0x01ED876150555E37,
    0xFF04876150535C37,0xFF05876150536E37,0xFF04876150655C37,0xFF05876150656E37,
    0xFF20876150537837,0xFF20876150657837,0x01F4876150554770,0xFF01876150535C70,
    0xFF17876150536E70,0xFF23876150534570,0xFF01876150655C70,0xFF17876150656E70,
    0xFF23876150654570,0x01FB876150551C72,0xFF0287615053457F,0xFFA8876150537870,
    0xFFA587615065457F,0xFF02876150530D7F,0xFFA8876150657870,0xFFA5876150650D7F,
    0x0202876150555E7F,0xFF02876150535C7F,0xFF02876150536E7F,0xFF25876150655C7F,
    0xFF26876150656E7F,0xFF0287615053787F,0xFF2687615065787F,0x0209418850551C1C,
    0x020A768850551C38,0x020B768850555D37,0xFF05738850536E37,0xFF05798850646E37,
    0xFF05808850646E37,0xFF05848850646E37,0xFFB1808850785C37,0xFFB1848850785C37,
    0x0212768850654638,0xFF04798850645C37,0xFF01738850654556,0xFF04808850645C37,
    0xFF04738850655C37,0xFF04848850645C37,0xFFB1798850785C37,0x0219768850551156,
    0xFF01738850534556,0xFF01798850644556,0xFF01808850644556,0xFF01848850644556,
    0xFFB9808850780D56,0xFFB9848850780D56,0x0220768850556F37,0xFF05738850656E37,
    0xFF20738850537837,0xFF20798850647837,0xFFB1798850786E37,0xFFB1808850786E37,
    0xFFB1848850786E37,0x0227768850657837,0xFF20808850647837,0xFF20738850657837,
    0xFF20848850647837,0xFF20798850787837,0xFF20808850787837,0xFF20848850787837,
    0x022E768850550D56,0xFF81738850530D56,0xFF81798850640D56,0xFF81808850640D56,
    0xFF81738850650D56,0xFF81848850640D56,0xFFB9798850780D56,0x0235768850551C57,
    0x0236768850555C56,0xFF04738850535C56,0xFF04798850645C56,0xFF04808850645C56,
    0xFF04738850655C56,0xFF04848850645C56,0xFF04798850785C56,0x023D768850555D56,
    0xFF9B738850536E56,0xFF9B798850646E56,0xFF9B808850646E56,0xFF9B848850646E56,
    0xFF04808850785C56,0xFF04848850785C56,0x0244768850556F56,0xFF9B738850656E56,
    0xFF1D738850537856,0xFF1D798850647856,0xFF1B798850786E56,0xFF1B808850786E56,
    0xFF1B848850786E56,0x024B768850551157,0xFF23738850534570,0xFFB9798850784556,
    0xFFB9808850784556,0xFFB9848850784556,0xFF24808850780D70,0xFF24848850780D70,
    0x0252768850657856,0xFF1D808850647856,0xFF1D738850657856,0xFF1D848850647856,
    0xFF1D798850787856,0xFF1D808850787856,0xFF1D848850787856,0x0259768850550D70,
    0xFF23738850530D70,0xFF23798850640D70,0xFF23808850640D70,0xFF23738850650D70,
    0xFF23848850640D70,0xFF24798850780D70,0x0260778850551C10,0x0261778850554710,
    0xFF04738850535C37,0xFF04878850535C37,0xFFB1808850784537,0xFFB1848850784537,
    0xFF1E798850786E0A,0xFF1E878850656E0A,0x0268778850654537,0xFF01798850644537,
    0xFF01808850644537,0xFF01738850654537,0xFF01848850644537,0xFFB1798850784537,
    0xFF01878850654537,0x026F778850551137,0xFF01738850534537,0xFF01878850534537,
    0xFFB9798850780D37,0xFFB9808850780D37,0xFF01878850650D37,0xFFB9848850780D37,
    0x0276778850556F0A,0xFF0573885053780A,0xFF0579885064780A,0xFF0580885064780A,
    0xFF1E808850786E0A,0xFF0587885053780A,0xFF1E848850786E0A,0x027D77885065780A,
    0xFF0573885065780A,0xFF0584885064780A,0xFF1E79885078780A,0xFF1E80885078780A,
    0xFF0587885065780A,0xFF1E84885078780A,0x0284778850550D37,0xFF01738850530D37,
    0xFF01798850640D37,0xFF01808850640D37,0xFF01738850650D37,0xFF01848850640D37,
    0xFF01878850530D37,0x028B768850551C72,0x028C768850555C70,0xFF01738850535C70,
    0xFF01798850645C70,0xFF01808850645C70,0xFF01738850655C70,0xFF01848850645C70,
    0xFF04798850785C70,0x0293768850555D70,0xFF17738850536E70,0xFF17798850646E70,
    0xFF17808850646E70,0xFF17848850646E70,0xFF04808850785C70,0xFF04848850785C70,
    0x029A768850654570,0xFF23808850644570,0xFF23738850654570,0xFF23848850644570,
    0xFF24798850784570,0xFF24808850784570,0xFF24848850784570,0x02A1768850556F70,
    0xFF17738850656E70,0xFFA8738850537870,0xFFA8798850647870,0xFF04798850786E70,
    0xFF04808850786E70,0xFF04848850786E70,0x02A8768850551172,0xFF23798850644570,
    0xFF02738850530D7F,0xFFA5798850640D7F,0xFFA5808850640D7F,0xFFA5738850650D7F,
    0xFFA5848850640D7F,0x02AF768850657870,0xFFA8808850647870,0xFFA8738850657870,
    0xFFA8848850647870,0xFF15798850787870,0xFF15808850787870,0xFF15848850787870,
    0x02B677885055170A,0x02B7778850655C0A,0xFF1E738850655C0A,0xFF1E848850645C0A,
    0xFF0C798850785C0A,0xFF0C808850785C0A,0xFF1E878850655C0A,0xFF0C848850785C0A,
    0x02BE77885055460A,0xFF1E738850535C0A,0xFF1E798850645C0A,0xFF1E808850645C0A,
    0xFF1E878850535C0A,0xFF0C80885078450A,0xFF0C84885078450A,0x02C5778850556E0A,
    0xFF1E738850536E0A,0xFF1E798850646E0A,0xFF1E808850646E0A,0xFF1E738850656E0A,
    0xFF1E848850646E0A,0xFF1E878850536E0A,0x02CC77885065450A,0xFF0C79885064450A,
    0xFF0C80885064450A,0xFF0C73885065450A,0xFF0C84885064450A,0xFF0C79885078450A,
    0xFF0C87885065450A,0x02D377885055110A,0xFF0C73885053450A,0xFF0C87885053450A,
    0xFF8C798850780D0A,0xFF8C808850780D0A,0xFF8C878850650D0A,0xFF8C848850780D0A,
    0x02DA778850550D0A,0xFF8C738850530D0A,0xFF8C798850640D0A,0xFF8C808850640D0A,
    0xFF8C738850650D0A,0xFF8C848850640D0A,0xFF8C878850530D0A,0x02E1408850551C7F,
    0x02E2768850655C7F,0xFF25798850645C7F,0xFF25808850645C7F,0xFF25738850655C7F,
    0xFF25848850645C7F,0xFF25798850785C7F,0xFF25808850785C7F,0x02E9768850555D7F,
    0xFF02738850536E7F,0xFF26798850646E7F,0xFF26808850646E7F,0xFF26738850656E7F,
    0xFF26848850646E7F,0xFF25848850785C7F,0x02F076885055467F,0xFF02738850535C7F,
    0xFFA573885065457F,0xFFA584885064457F,0xFFA579885078457F,0xFFA580885078457F,
    0xFFA584885078457F,0x02F776885055117F,0xFF0273885053457F,0xFFA579885064457F,
    0xFFA580885064457F,0xFFA5798850780D7F,0xFFA5808850780D7F,0xFFA5848850780D7F,
    0x02FE768850556F7F,0xFF0273885053787F,0xFF2679885064787F,0xFF26798850786E7F,
    0xFF2680885064787F,0xFF26808850786E7F,0xFF26848850786E7F,0x030540885065787F,
    0xFF263B885065787F,0xFF2673885065787F,0xFF2684885064787F,0xFF2679885078787F,
    0xFF2680885078787F,0xFF2684885078787F,0x030C0E8850551C1C,0x030D208850551C38,
    0x030E208850655C56,0xFF042E8850645C56,0xFF04278850645C56,0xFF043B8850655C56,
    0xFFB12E8850785C56,0xFFB1278850785C56,0xFFB11D8850785C56,0x0315208850554656,
    0xFF043B8850535C56,0xFF041D8850645C56,0xFF013B8850654556,0xFFB12E8850784556,
    0xFFB1278850784556,0xFFB11D8850784556,0x031C208850546F38,0xFF1B3B8850536E56,
    0xFF1B2E8850646E56,0xFF1B278850646E56,0xFF1B1D8850646E56,0xFFA03B8850537837,
    0xFFA01D8850647837,0x0323208850551156,0xFF013B8850534556,0xFF012E8850644556,
    0xFF01278850644556,0xFF011D8850644556,0xFFB12E8850780D56,0xFFB1278850780D56,
    0x032A208850657837,0xFFA02E8850647837,0xFFA0278850647837,0xFFA03B8850657837,
    0xFFA02E8850787837,0xFFA0278850787837,0xFFA01D8850787837,0x0331208850550D56,
    0xFF843B8850530D56,0xFF842E8850640D56,0xFF84278850640D56,0xFF843B8850650D56,
    0xFF841D8850640D56,0xFFB11D8850780D56,0x03380E8850551C57,0x0339208850556F56,
    0xFF1B3B8850656E56,0xFF1D3B8850537856,0xFFB12E8850786E56,0xFFB1278850786E56,
    0xFF1D1D8850647856,0xFFB11D8850786E56,0x0340208850657856,0xFF1D2E8850647856,
    0xFF1D278850647856,0xFF1D3B8850657856,0xFF1D2E8850787856,0xFF1D278850787856,
    0xFF1D1D8850787856,0x03470E8850555C70,0xFF043B8850535C70,0xFF042E8850645C70,
    0xFF04278850645C70,0xFF043B8850655C70,0xFF041D8850645C70,0xFF04068850655C70,
    0x034E0E8850554670,0xFF04068850535C70,0xFF233B8850654570,0xFF242E8850784570,
    0xFF24278850784570,0xFF23068850654570,0xFF241D8850784570,0x0355208850551170,
    0xFF233B8850534570,0xFF232E8850644570,0xFF23278850644570,0xFF231D8850644570,
    0xFF242E8850780D70,0xFF24278850780D70,0x035C208850550D70,0xFF233B8850530D70,
    0xFF232E8850640D70,0xFF23278850640D70,0xFF233B8850650D70,0xFF231D8850640D70,
    0xFF241D8850780D70,0x0363208850551C10,0x0364208850655C37,0xFF042E8850645C37,
    0xFF04278850645C37,0xFF043B8850655C37,0xFF042E8850785C37,0xFF04278850785C37,
    0xFF041D8850785C37,0x036B208850554637,0xFF043B8850535C37,0xFF041D8850645C37,
    0xFF013B8850654537,0xFFB12E8850784537,0xFFB1278850784537,0xFFB11D8850784537,
    0x0372208850556E37,0xFF053B8850536E37,0xFF052E8850646E37,0xFF05278850646E37,
    0xFF053B8850656E37,0xFF051D8850646E37,0xFF051D8850786E37,0x0379208850551137,
    0xFF013B8850534537,0xFF012E8850644537,0xFF01278850644537,0xFF011D8850644537,
    0xFFB12E8850780D37,0xFFB1278850780D37,0x0380208850656F10,0xFF052E8850786E37,
    0xFF05278850786E37,0xFF053B885065780A,0xFF1E2E885078780A,0xFF1E27885078780A,
    0xFF1E1D885078780A,0x0387208850550D37,0xFF013B8850530D37,0xFF012E8850640D37,
    0xFF01278850640D37,0xFF013B8850650D37,0xFF011D8850640D37,0xFFB11D8850780D37,
    0x038E0E8850551C72,0x038F0E8850554772,0xFF153B8850536E70,0xFF042E8850785C70,
    0xFF15068850536E70,0xFF04278850785C70,0xFF041D8850785C70,0xFF251D885064457F,
    0x03960E8850656E70,0xFF152E8850646E70,0xFF15278850646E70,0xFF153B8850656E70,
    0xFF151D8850646E70,0xFF15068850656E70,0xFF041D8850786E70,0x039D0E8850556F70,
    0xFF153B8850537870,0xFF042E8850786E70,0xFF15278850647870,0xFF04278850786E70,
    0xFF151D8850647870,0xFF15068850537870,0x03A40E885055117F,0xFF023B885053457F,
    0xFF0206885053457F,0xFF253B8850650D7F,0xFF252E8850780D7F,0xFF25278850780D7F,
    0xFF251D8850780D7F,0x03AB0E8850657870,0xFF152E8850647870,0xFF153B8850657870,
    0xFF152E8850787870,0xFF15278850787870,0xFF15068850657870,0xFF151D8850787870,
    0x03B20E8850550D7F,0xFF023B8850530D7F,0xFF252E8850640D7F,0xFF25278850640D7F,
    0xFF251D8850640D7F,0xFF02068850530D7F,0xFF25068850650D7F,0x03B9208850551C0A,
    0x03BA208850655C0A,0xFF0C2E8850645C0A,0xFF0C278850645C0A,0xFF0C3B8850655C0A,
    0xFF0C2E8850785C0A,0xFF0C278850785C0A,0xFF0C1D8850785C0A,0x03C120885055460A,
    0xFF0C3B8850535C0A,0xFF0C1D8850645C0A,0xFF0C3B885065450A,0xFF0C2E885078450A,
    0xFF0C27885078450A,0xFF0C1D885078450A,0x03C8208850556E0A,0xFF1E3B8850536E0A,
    0xFF1E2E8850646E0A,0xFF1E278850646E0A,0xFF1E3B8850656E0A,0xFF1E1D8850646E0A,
    0xFF1E1D8850786E0A,0x03CF20885055110A,0xFF0C3B885053450A,0xFF0C2E885064450A,
    0xFF0C27885064450A,0xFF0C1D885064450A,0xFF0C2E8850780D0A,0xFF0C278850780D0A,
    0x03D6208850556F0A,0xFF053B885053780A,0xFF052E885064780A,0xFF1E2E8850786E0A,
    0xFF0527885064780A,0xFF1E278850786E0A,0xFF051D885064780A,0x03DD208850550D0A,
    0xFF0C3B8850530D0A,0xFF0C2E8850640D0A,0xFF0C278850640D0A,0xFF0C3B8850650D0A,
    0xFF0C1D8850640D0A,0xFF0C1D8850780D0A,0x03E40E885055487F,0x03E50E8850555D7F,
    0xFF253B8850655C7F,0xFF252E8850785C7F,0xFF02068850536E7F,0xFF25278850785C7F,
    0xFF25068850655C7F,0xFF251D8850785C7F,0x03EC0E885055467F,0xFF023B8850535C7F,
    0xFF252E8850645C7F,0xFF25278850645C7F,0xFF251D8850645C7F,0xFF02068850535C7F,
    0xFF252E885078457F,0x03F30E8850556E7F,0xFF023B8850536E7F,0xFF262E8850646E7F,
    0xFF26278850646E7F,0xFF263B8850656E7F,0xFF261D8850646E7F,0xFF26068850656E7F,
    0x03FA0E885065457F,0xFF252E885064457F,0xFF2527885064457F,0xFF253B885065457F,
    0xFF2527885078457F,0xFF2506885065457F,0xFF251D885078457F,0x04010E8850556F7F,
    0xFF023B885053787F,0xFF262E8850786E7F,0xFF26278850786E7F,0xFF261D885064787F,
    0xFF0206885053787F,0xFF261D8850786E7F,0x04080B885065787F,0xFF262E885064787F,
    0xFF2627885064787F,0xFF262E885078787F,0xFF2627885078787F,0xFF2606885065787F,
    0xFF261D885078787F,0x040F236168551C1C,0x0410236168551C38,0x0411236168555D38,
    0xFF103B6168525C56,0xFF04276168555C56,0xFF05736168556E37,0xFF041D6168555C56,
    0xFF052E6168556E37,0xFF05796168556E37,0x0418346168554556,0xFF014961685B4556,
    0xFF016661685B4556,0xFF01736168554556,0xFF013B61685B4556,0xFF012E6168554556,
    0xFF01796168554556,0x041F216168551156,0xFF103B6168524556,0xFF844961685B0D56,
    0xFF816661685B0D56,0xFF01276168554556,0xFF843B61685B0D56,0xFF011D6168554556,
    0x0426216168556F37,0xFF054961685B6E37,0xFF056661685B6E37,0xFF053B61685B6E37,
    0xFF103B6168527837,0xFFA0276168557837,0xFFA01D6168557837,0x042D346168557837,
    0xFFA04961685B7837,0xFF206661685B7837,0xFF20736168557837,0xFFA03B61685B7837,
    0xFFA02E6168557837,0xFF20796168557837,0x0434236168550D56,0xFF103B6168520D56,
    0xFF81736168550D56,0xFF842E6168550D56,0xFF81796168550D56,0xFF84276168550D56,
    0xFF841D6168550D56,0x043B236168551C57,0x043C346168555C56,0xFF044961685B5C56,
    0xFF046661685B5C56,0xFF04736168555C56,0xFF043B61685B5C56,0xFF042E6168555C56,
    0xFF04796168555C56,0x0443236168556E56,0xFF103B6168526E56,0xFF9B736168556E56,
    0xFF1B2E6168556E56,0xFF9B796168556E56,0xFF1B276168556E56,0xFF1B1D6168556E56,
    0x044A216168556F56,0xFF1B4961685B6E56,0xFF9B6661685B6E56,0xFF1B3B61685B6E56,
    0xFF103B6168527856,0xFF1D276168557856,0xFF1D1D6168557856,0x0451346168557856,
    0xFF1D4961685B7856,0xFF1D6661685B7856,0xFF1D736168557856,0xFF1D3B61685B7856,
    0xFF1D2E6168557856,0xFF1D796168557856,0x0458216168551170,0xFF103B6168524570,
    0xFF234961685B0D70,0xFF236661685B0D70,0xFF23276168554570,0xFF233B61685B0D70,
    0xFF231D6168554570,0x045F236168550D70,0xFF103B6168520D70,0xFF23736168550D70,
    0xFF232E6168550D70,0xFF23796168550D70,0xFF23276168550D70,0xFF231D6168550D70,
    0x0466236168551C10,0x0467236168555C37,0xFF103B6168525C37,0xFF04736168555C37,
    0xFF042E6168555C37,0xFF04796168555C37,0xFF04276168555C37,0xFF041D6168555C37,
    0x046E216168555D37,0xFF044961685B5C37,0xFF046661685B5C37,0xFF103B6168526E37,
    0xFF043B61685B5C37,0xFF05276168556E37,0xFF051D6168556E37,0x0475346168554537,
    0xFF014961685B4537,0xFF016661685B4537,0xFF01736168554537,0xFF013B61685B4537,
    0xFF012E6168554537,0xFF01796168554537,0x047C216168551137,0xFF103B6168524537,
    0xFF014961685B0D37,0xFF016661685B0D37,0xFF01276168554537,0xFF013B61685B0D37,
    0xFF011D6168554537,0x0483236168550D37,0xFF103B6168520D37,0xFF01736168550D37,
    0xFF012E6168550D37,0xFF01796168550D37,0xFF01276168550D37,0xFF011D6168550D37,
    0x048A34616855780A,0xFF054961685B780A,0xFF056661685B780A,0xFF0573616855780A,
    0xFF053B61685B780A,0xFF052E616855780A,0xFF0579616855780A,0x0491236168551C72,
    0x0492236168555C70,0xFF103B6168525C70,0xFF01736168555C70,0xFF042E6168555C70,
    0xFF01796168555C70,0xFF04276168555C70,0xFF041D6168555C70,0x0499216168555D70,
    0xFF044961685B5C70,0xFF016661685B5C70,0xFF103B6168526E70,0xFF043B61685B5C70,
    0xFF15276168556E70,0xFF151D6168556E70,0x04A0346168556E70,0xFF154961685B6E70,
    0xFF176661685B6E70,0xFF17736168556E70,0xFF153B61685B6E70,0xFF152E6168556E70,
    0xFF17796168556E70,0x04A7346168554570,0xFF234961685B4570,0xFF236661685B4570,
    0xFF23736168554570,0xFF233B61685B4570,0xFF232E6168554570,0xFF23796168554570,
    0x04AE226168557870,0xFF103B6168527870,0xFFA8736168557870,0xFF153B61685B7870,
    0xFF152E6168557870,0xFF15276168557870,0xFF151D6168557870,0x04B5216168551C72,
    0xFF154961685B7870,0xFFA86661685B7870,0xFF023B6168520D7F,0xFF022E6168550D7F,
    0xFF02276168550D7F,0xFF021D6168550D7F,0x04BC236168551C0A,0x04BD346168555C0A,
    0xFF0C4961685B5C0A,0xFF1E6661685B5C0A,0xFF1E736168555C0A,0xFF0C3B61685B5C0A,
    0xFF0C2E6168555C0A,0xFF1E796168555C0A,0x04C42D616855460A,0xFF1A3B6168525C0A,
    0xFF0C4961685B450A,0xFF0C6661685B450A,0xFF0C3B61685B450A,0xFF0C276168555C0A,
    0xFF0C79616855450A,0x04CB236168556E0A,0xFF1A3B6168526E0A,0xFF1E736168556E0A,
    0xFF1E2E6168556E0A,0xFF1E796168556E0A,0xFF1E276168556E0A,0xFF1E1D6168556E0A,
    0x04D22C616855110A,0xFF1A3B616852450A,0xFF0C73616855450A,0xFF0C2E616855450A,
    0xFF0C4961685B0D0A,0xFF8C6661685B0D0A,0xFF0C27616855450A,0x04D9216168556F0A,
    0xFF1E4961685B6E0A,0xFF1E6661685B6E0A,0xFF1E3B61685B6E0A,0xFF1A3B616852780A,
    0xFF0527616855780A,0xFF051D616855780A,0x04E02D6168550D0A,0xFF1A3B6168520D0A,
    0xFF8C736168550D0A,0xFF0C3B61685B0D0A,0xFF0C2E6168550D0A,0xFF8C796168550D0A,
    0xFF0C276168550D0A,0x04E7226168551C7F,0x04E8226168555C7F,0xFF023B6168525C7F,
    0xFF02736168555C7F,0xFF023B61685B5C7F,0xFF022E6168555C7F,0xFF02276168555C7F,
    0xFF021D6168555C7F,0x04EF216168555D7F,0xFF024961685B5C7F,0xFF026661685B5C7F,
    0xFF023B6168526E7F,0xFF022E6168556E7F,0xFF02276168556E7F,0xFF021D6168556E7F,
    0x04F62C616855457F,0xFF024961685B457F,0xFF026661685B457F,0xFF0273616855457F,
    0xFF023B61685B457F,0xFF022E616855457F,0xFF0227616855457F,0x04FD22616855117F,
    0xFF023B616852457F,0xFF024961685B0D7F,0xFF026661685B0D7F,0xFF02736168550D7F,
    0xFF023B61685B0D7F,0xFF021D616855457F,0x0504226168556F7F,0xFF024961685B6E7F,
    0xFF026661685B6E7F,0xFF02736168556E7F,0xFF023B61685B6E7F,0xFF023B616852787F,
    0xFF021D616855787F,0x050B2C616855787F,0xFF024961685B787F,0xFF026661685B787F,
    0xFF0273616855787F,0xFF023B61685B787F,0xFF022E616855787F,0xFF0227616855787F,
    0x05121C6268551C1C,0x0513356268551C1C,0x0514796168551C72,0xFF02796168555C7F,
    0xFF02796168556E7F,0xFF0279616855457F,0xFFA8796168557870,0xFF0279616855787F,
    0xFF02796168550D7F,0x051B356268551C15,0xFF1E806168555C0A,0xFF0C80616855450A,
    0xFF052E8868556E37,0xFF8C806168550D0A,0xFF842E8868550D56,0xFFA02E8868557837,
    0x05223B8868555E0A,0xFF1A3B8868525C0A,0xFF1A3B8868526E0A,0xFF0C3B88685B5C0A,
    0xFF1E3B88685B6E0A,0xFF1A3B886852780A,0xFF053B88685B780A,0x05292F8868551C1C,
    0xFF1A3B886852450A,0xFF0C3B88685B450A,0xFF022E8868556E7F,0xFF1A3B8868520D0A,
    0xFF0C3B88685B0D0A,0xFF022E886855787F,0x05302E8868551C57,0xFF042E8868555C56,
    0xFF1B2E8868556E56,0xFF012E8868554556,0xFF232E8868554570,0xFF1D2E8868557856,
    0xFF232E8868550D70,0x05372E8868551C72,0xFF042E8868555C70,0xFF152E8868556E70,
    0xFF022E8868555C7F,0xFF022E886855457F,0xFF152E8868557870,0xFF022E8868550D7F,
    0x053E1F8868551C1C,0x053F278868551C57,0xFF04278868555C56,0xFF1B278868556E56,
    0xFF01278868554556,0xFF1D278868557856,0xFF84278868550D56,0xFF23278868550D70,
    0x0546288868554837,0xFF042E8868555C37,0xFF04278868555C37,0xFF012E8868554537,
    0xFF05278868556E37,0xFF01278868554537,0xFFA0278868557837,0x054D288868551C10,
    0xFF1E2E8868556E0A,0xFF012E8868550D37,0xFF1E278868556E0A,0xFF01278868550D37,
    0xFF052E886855780A,0xFF0527886855780A,0x0554278868551C72,0xFF04278868555C70,
    0xFF15278868556E70,0xFF23278868554570,0xFF0227886855457F,0xFF15278868557870,
    0xFF02278868550D7F,0x055B28886855140A,0xFF0C2E8868555C0A,0xFF0C278868555C0A,
    0xFF0C2E886855450A,0xFF0C27886855450A,0xFF0C2E8868550D0A,0xFF0C278868550D0A,
    0x05621E8868555E7F,0xFF02278868555C7F,0xFF021D8868555C7F,0xFF02278868556E7F,
    0xFF021D8868556E7F,0xFF0227886855787F,0xFF021D886855787F,0x0569816168551C1C,
    0x056A816168555D15,0xFF04846168555C56,0xFF9B806168556E56,0xFF04846168555C37,
    0xFF05806168556E37,0xFF1E846168555C0A,0xFF1E806168556E0A,0x057181616855463A,
    0xFF04806168555C56,0xFF04806168555C37,0xFF01806168555C70,0xFF01846168554556,
    0xFF01846168554537,0xFF02806168555C7F,0x057881616855111C,0xFF01806168554556,
    0xFF01806168554537,0xFF23806168554570,0xFF81846168550D56,0xFF0280616855457F,
    0xFF0C84616855450A,0x057F816168556F1C,0xFF9B846168556E56,0xFF17806168556E70,
    0xFF05846168556E37,0xFF02806168556E7F,0xFF1E846168556E0A,0xFF0580616855780A,
    0x058681616855781C,0xFF1D806168557856,0xFF20806168557837,0xFFA8806168557870,
    0xFF20846168557837,0xFF0280616855787F,0xFF0584616855780A,0x058D816168550D1C,
    0xFF81806168550D56,0xFF01806168550D37,0xFF23806168550D70,0xFF01846168550D37,
    0xFF02806168550D7F,0xFF8C846168550D0A,0x0594856168551C1C,0x0595876168551C38,
    0xFF01876168554556,0xFF04876168555C37,0xFF05876168556E37,0xFF01876168554537,
    0xFF81876168550D56,0xFF20876168557837,0x059C856168551C57,0xFF04876168555C56,
    0xFF9B876168556E56,0xFF1D846168557856,0xFF1D876168557856,0xFF23846168550D70,
    0xFF23876168550D70,0x05A3856168554770,0xFF01846168555C70,0xFF01876168555C70,
    0xFF17846168556E70,0xFF23846168554570,0xFF17876168556E70,0xFF23876168554570,
    0x05AA876168551C10,0xFF1E876168555C0A,0xFF1E876168556E0A,0xFF0C87616855450A,
    0xFF01876168550D37,0xFF0587616855780A,0xFF8C876168550D0A,0x05B1856168551C72,
    0xFF0284616855457F,0xFFA8846168557870,0xFF0287616855457F,0xFFA8876168557870,
    0xFF02846168550D7F,0xFF02876168550D7F,0x05B8856168555E7F,0xFF02846168555C7F,
    0xFF02876168555C7F,0xFF02846168556E7F,0xFF02876168556E7F,0xFF0284616855787F,
    0xFF0287616855787F,0x05BF088868551C1C,0x05C01D8868551C38,0xFF011D8868554556,
    0xFF041D8868555C37,0xFF051D8868556E37,0xFF011D8868554537,0xFF841D8868550D56,
    0xFFA01D8868557837,0x05C71D8868551C57,0xFF041D8868555C56,0xFF1B1D8868556E56,
    0xFF041D8868555C70,0xFF231D8868554570,0xFF1D1D8868557856,0xFF231D8868550D70,
    0x05CE088868556F70,0xFF10068868526E70,0xFF151D8868556E70,0xFF150688685B6E70,
    0xFF10068868527870,0xFF151D8868557870,0xFF150688685B7870,0x05D51D8868551C10,
    0xFF0C1D8868555C0A,0xFF1E1D8868556E0A,0xFF0C1D886855450A,0xFF011D8868550D37,
    0xFF051D886855780A,0xFF0C1D8868550D0A,0x05DC068868555E7F,0xFF02068868525C7F,
    0xFF02068868526E7F,0xFF020688685B5C7F,0xFF020688685B6E7F,0xFF0206886852787F,
    0xFF020688685B787F,0x05E308886855117F,0xFF0206886852457F,0xFF021D886855457F,
    0xFF020688685B457F,0xFF02068868520D7F,0xFF021D8868550D7F,0xFF020688685B0D7F,
    0x05EA068868551C19,0x05EB068868555E56,0xFF10068868525C56,0xFF10068868526E56,
    0xFF040688685B5C56,0xFF1B0688685B6E56,0xFF10068868527856,0xFF1D0688685B7856,
    0x05F2068868551C38,0xFF10068868524556,0xFF010688685B4556,0xFF10068868520D56,
    0xFF10068868527837,0xFF840688685B0D56,0xFFA00688685B7837,0x05F9068868554737,
    0xFF10068868525C37,0xFF10068868526E37,0xFF10068868524537,0xFF040688685B5C37,
    0xFF050688685B6E37,0xFF010688685B4537,0x0600068868551470,0xFF10068868525C70,
    0xFF10068868524570,0xFF040688685B5C70,0xFF230688685B4570,0xFF10068868520D70,
    0xFF230688685B0D70,0x0607068868551C10,0xFF1A068868526E0A,0xFF10068868520D37,
    0xFF1E0688685B6E0A,0xFF010688685B0D37,0xFF1A06886852780A,0xFF050688685B780A,
    0x060E06886855140A,0xFF1A068868525C0A,0xFF1A06886852450A,0xFF0C0688685B5C0A,
    0xFF0C0688685B450A,0xFF1A068868520D0A,0xFF0C0688685B0D0A,0x06151C6243551C1C,
    0x0616336143551C1C,0x0617326143551C38,0x0618326143555C56,0xFF04496143595C56,
    0xFF04666143595C56,0xFF043B6150535C56,0xFF042E6143535C56,0xFF042E6150645C56,
    0xFF04496143655C56,0x061F3261435B4556,0xFF01666143594556,0xFF012E6150644556,
    0xFF01496143654556,0xFF01666143654556,0xFF013B6150654556,0xFFB12E6150784556,
    0x0626326143555E38,0xFF04666143655C56,0xFF043B6150655C56,0xFFB12E6150785C56,
    0xFFA0496143597837,0xFFA03B6150537837,0xFFA02E6143537837,0x062D326143551156,
    0xFF01496143594556,0xFF013B6150534556,0xFF012E6143534556,0xFF81666143650D56,
    0xFF843B6150650D56,0xFFB12E6150780D56,0x06343261435B7837,0xFF20666143597837,
    0xFFA02E6150647837,0xFFA0496143657837,0xFF20666143657837,0xFFA03B6150657837,
    0xFFA02E6150787837,0x063B326143550D56,0xFF84496143590D56,0xFF81666143590D56,
    0xFF843B6150530D56,0xFF842E6143530D56,0xFF842E6150640D56,0xFF84496143650D56,
    0x0642326143551C57,0x0643326143556E56,0xFF1B496143596E56,0xFF9B666143596E56,
    0xFF1B3B6150536E56,0xFF1B2E6143536E56,0xFF1B2E6150646E56,0xFF1B496143656E56,
    0x064A326143556F56,0xFF1D496143597856,0xFF9B666143656E56,0xFF1B3B6150656E56,
    0xFF1D3B6150537856,0xFF1D2E6143537856,0xFFB12E6150786E56,0x06513261435B7856,
    0xFF1D666143597856,0xFF1D2E6150647856,0xFF1D496143657856,0xFF1D666143657856,
    0xFF1D3B6150657856,0xFF1D2E6150787856,0x06583261435B4570,0xFF23666143594570,
    0xFF232E6150644570,0xFF23496143654570,0xFF23666143654570,0xFF233B6150654570,
    0xFF242E6150784570,0x065F326143551170,0xFF23496143594570,0xFF233B6150534570,
    0xFF232E6143534570,0xFF23666143650D70,0xFF233B6150650D70,0xFF242E6150780D70,
    0x0666326143550D70,0xFF23496143590D70,0xFF23666143590D70,0xFF233B6150530D70,
    0xFF232E6143530D70,0xFF232E6150640D70,0xFF23496143650D70,0x066D326143551C10,
    0x066E4B61435B1737,0xFF04496143595C37,0xFF05496143596E37,0xFF04496143655C37,
    0xFF05496143656E37,0xFF01666143590D37,0xFF01666143650D37,0x06756661435B4737,
    0xFF04666143595C37,0xFF05666143596E37,0xFF01666143594537,0xFF04666143655C37,
    0xFF05666143656E37,0xFF01666143654537,0x067C3C6143551737,0xFF01496143594537,
    0xFF053B6150536E37,0xFF01496143590D37,0xFF01496143654537,0xFF053B6150656E37,
    0xFF01496143650D37,0x06833B6150551437,0xFF043B6150535C37,0xFF013B6150534537,
    0xFF043B6150655C37,0xFF013B6150654537,0xFF013B6150530D37,0xFF013B6150650D37,
    0x068A2F6143555E10,0xFF042E6150645C37,0xFF052E6143536E37,0xFF052E6150646E37,
    0xFF042E6150785C37,0xFF052E6150786E37,0xFF053B615065780A,0x06912E6143551437,
    0xFF042E6143535C37,0xFF012E6143534537,0xFF012E6150644537,0xFF012E6150640D37,
    0xFFB12E6150784537,0xFFB12E6150780D37,0x0698326143551C72,0x06993261435B5C70,
    0xFF01666143595C70,0xFF042E6150645C70,0xFF04496143655C70,0xFF01666143655C70,
    0xFF043B6150655C70,0xFF042E6150785C70,0x06A0326143556E70,0xFF15496143596E70,
    0xFF17666143596E70,0xFF153B6150536E70,0xFF152E6143536E70,0xFF152E6150646E70,
    0xFF15496143656E70,0x06A7326143551472,0xFF04496143595C70,0xFF043B6150535C70,
    0xFF042E6143535C70,0xFFA5666143650D7F,0xFF253B6150650D7F,0xFF252E6150780D7F,
    0x06AE326143556F70,0xFF15496143597870,0xFF17666143656E70,0xFF153B6150656E70,
    0xFF153B6150537870,0xFF152E6143537870,0xFF042E6150786E70,0x06B53261435B7870,
    0xFFA8666143597870,0xFF152E6150647870,0xFF15496143657870,0xFFA8666143657870,
    0xFF153B6150657870,0xFF152E6150787870,0x06BC326143550D7F,0xFF02496143590D7F,
    0xFF02666143590D7F,0xFF023B6150530D7F,0xFF022E6143530D7F,0xFF252E6150640D7F,
    0xFF25496143650D7F,0x06C33E6143551C0A,0x06C43E61435B5C0A,0xFF0C496143595C0A,
    0xFF1E666143595C0A,0xFF0C496143655C0A,0xFF1E666143655C0A,0xFF0C3B6150655C0A,
    0xFF1E736150655C0A,0x06CB3E614355460A,0xFF0C3B6150535C0A,0xFF1E736150535C0A,
    0xFF0C49614365450A,0xFF0C66614365450A,0xFF0C3B615065450A,0xFF0C73615065450A,
    0x06D23E6143556E0A,0xFF1E496143596E0A,0xFF1E666143596E0A,0xFF1E3B6150536E0A,
    0xFF1E736150536E0A,0xFF1E496143656E0A,0xFF1E666143656E0A,0x06D93E614355110A,
    0xFF0C49614359450A,0xFF0C66614359450A,0xFF0C3B615053450A,0xFF0C73615053450A,
    0xFF0C3B6150650D0A,0xFF8C736150650D0A,0x06E03D6143556F0A,0xFF0549614359780A,
    0xFF0566614359780A,0xFF1E3B6150656E0A,0xFF053B615053780A,0xFF0549614365780A,
    0xFF0566614365780A,0x06E73E6143550D0A,0xFF0C496143590D0A,0xFF8C666143590D0A,
    0xFF0C3B6150530D0A,0xFF8C736150530D0A,0xFF0C496143650D0A,0xFF8C666143650D0A,
    0x06EE32614355487F,0x06EF4B61435B487F,0xFF02496143596E7F,0xFF0266614359457F,
    0xFF0249614359787F,0xFF26496143656E7F,0xFFA566614365457F,0xFF2649614365787F,
    0x06F66661435B5E7F,0xFF02666143595C7F,0xFF02666143596E7F,0xFF25666143655C7F,
    0xFF0266614359787F,0xFF26666143656E7F,0xFF2666614365787F,0x06FD3C614355487F,
    0xFF02496143595C7F,0xFF0249614359457F,0xFF25496143655C7F,0xFF2549614365457F,
    0xFF023B615053787F,0xFF263B615065787F,0x07043B615055477F,0xFF023B6150535C7F,
    0xFF023B6150536E7F,0xFF023B615053457F,0xFF253B6150655C7F,0xFF263B6150656E7F,
    0xFF253B615065457F,0x070B2E614355467F,0xFF022E6143535C7F,0xFF252E6150645C7F,
    0xFF022E614353457F,0xFF252E615064457F,0xFF252E6150785C7F,0xFF252E615078457F,
    0x07122E6143556F7F,0xFF022E6143536E7F,0xFF262E6150646E7F,0xFF022E614353787F,
    0xFF262E615064787F,0xFF262E6150786E7F,0xFF262E615078787F,0x07190B6143551C1C,
    0x071A096143551C38,0x071B096143551156,0xFF01276143534556,0xFF011D6143534556,
    0xFF011D6150644556,0xFF01066150534556,0xFFB1276150780D56,0xFFB11D6150780D56,
    0x0722096143550D56,0xFF84276143530D56,0xFF84276150640D56,0xFF841D6143530D56,
    0xFF841D6150640D56,0xFF84066150530D56,0xFF84066150650D56,0x0729096143555C37,
    0xFF04276143535C37,0xFF04276150645C37,0xFF041D6143535C37,0xFF041D6150645C37,
    0xFF04066150535C37,0xFF04066150655C37,0x0730096143555D37,0xFF05276143536E37,
    0xFF051D6143536E37,0xFF051D6150646E37,0xFF05066150536E37,0xFF04276150785C37,
    0xFF041D6150785C37,0x0737096143556F37,0xFF05276150646E37,0xFFA0276143537837,
    0xFF05276150786E37,0xFFA01D6143537837,0xFF05066150656E37,0xFF051D6150786E37,
    0x073E096150557837,0xFFA0276150647837,0xFFA01D6150647837,0xFFA0066150537837,
    0xFFA0276150787837,0xFFA0066150657837,0xFFA01D6150787837,0x0745096143551C57,
    0x0746096143555C56,0xFF04276143535C56,0xFF04276150645C56,0xFF041D6143535C56,
    0xFF041D6150645C56,0xFF04066150535C56,0xFF04066150655C56,0x074D096143555D56,
    0xFF1B276143536E56,0xFF1B1D6143536E56,0xFF1B1D6150646E56,0xFF1B066150536E56,
    0xFFB1276150785C56,0xFFB11D6150785C56,0x0754096143556F56,0xFF1B276150646E56,
    0xFF1D276143537856,0xFFB1276150786E56,0xFF1D1D6143537856,0xFF1B066150656E56,
    0xFFB11D6150786E56,0x075B096150651157,0xFF01276150644556,0xFFB1276150784556,
    0xFF01066150654556,0xFFB11D6150784556,0xFF24276150780D70,0xFF241D6150780D70,
    0x0762096150557856,0xFF1D276150647856,0xFF1D1D6150647856,0xFF1D066150537856,
    0xFF1D276150787856,0xFF1D066150657856,0xFF1D1D6150787856,0x0769096143550D70,
    0xFF23276143530D70,0xFF23276150640D70,0xFF231D6143530D70,0xFF231D6150640D70,
    0xFF23066150530D70,0xFF23066150650D70,0x07700B6143551C10,0x0771096150554710,
    0xFF01276150644537,0xFF011D6150644537,0xFFB1276150784537,0xFF1E066150536E0A,
    0xFF01066150654537,0xFFB11D6150784537,0x07780B6150656E0A,0xFF1E2E6150646E0A,
    0xFF1E276150646E0A,0xFF1E1D6150646E0A,0xFF1E276150786E0A,0xFF1E066150656E0A,
    0xFF1E1D6150786E0A,0x077F0B6143556F0A,0xFF052E614353780A,0xFF1E2E6150786E0A,
    0xFF0527614353780A,0xFF051D614353780A,0xFF051D615064780A,0xFF0506615053780A,
    0x0786096143551137,0xFF01276143534537,0xFF011D6143534537,0xFF01066150534537,
    0xFFB1276150780D37,0xFF01066150650D37,0xFFB11D6150780D37,0x078D0B615065780A,
    0xFF052E615064780A,0xFF0527615064780A,0xFF1E2E615078780A,0xFF1E27615078780A,
    0xFF0506615065780A,0xFF1E1D615078780A,0x07940B6143540D37,0xFF012E6143530D37,
    0xFF01276143530D37,0xFF01276150640D37,0xFF011D6143530D37,0xFF011D6150640D37,
    0xFF01066150530D37,0x079B096143551C72,0x079C096143555C70,0xFF04276143535C70,
    0xFF04276150645C70,0xFF041D6143535C70,0xFF041D6150645C70,0xFF04066150535C70,
    0xFF04066150655C70,0x07A3096143555D70,0xFF15276143536E70,0xFF151D6143536E70,
    0xFF151D6150646E70,0xFF15066150536E70,0xFF04276150785C70,0xFF041D6150785C70,
    0x07AA096150554570,0xFF23276150644570,0xFF231D6150644570,0xFF23066150534570,
    0xFF24276150784570,0xFF23066150654570,0xFF241D6150784570,0x07B1096143541172,
    0xFF23276143534570,0xFF231D6143534570,0xFF02276143530D7F,0xFF021D6143530D7F,
    0xFF251D6150640D7F,0xFF02066150530D7F,0x07B8096143556F70,0xFF15276150646E70,
    0xFF15276143537870,0xFF04276150786E70,0xFF151D6143537870,0xFF15066150656E70,
    0xFF041D6150786E70,0x07BF096150557870,0xFF15276150647870,0xFF151D6150647870,
    0xFF15066150537870,0xFF15276150787870,0xFF15066150657870,0xFF151D6150787870,
    0x07C60B614355170A,0x07C708615054140A,0xFF0C1D6150645C0A,0xFF0C066150535C0A,
    0xFF0C1D615064450A,0xFF0C06615053450A,0xFF0C1D6150640D0A,0xFF0C066150530D0A,
    0x07CE28614353170A,0xFF0C2E6143535C0A,0xFF0C276143535C0A,0xFF1E2E6143536E0A,
    0xFF0C2E614353450A,0xFF1E276143536E0A,0xFF0C2E6143530D0A,0x07D51E614353170A,
    0xFF0C1D6143535C0A,0xFF0C27614353450A,0xFF1E1D6143536E0A,0xFF0C1D614353450A,
    0xFF0C276143530D0A,0xFF0C1D6143530D0A,0x07DC28615064140A,0xFF0C2E6150645C0A,
    0xFF0C276150645C0A,0xFF0C2E615064450A,0xFF0C27615064450A,0xFF0C2E6150640D0A,
    0xFF0C276150640D0A,0x07E308615065140A,0xFF0C066150655C0A,0xFF0C1D6150785C0A,
    0xFF0C06615065450A,0xFF0C1D615078450A,0xFF0C066150650D0A,0xFF0C1D6150780D0A,
    0x07EA28615078140A,0xFF0C2E6150785C0A,0xFF0C276150785C0A,0xFF0C2E615078450A,
    0xFF0C27615078450A,0xFF0C2E6150780D0A,0xFF0C276150780D0A,0x07F1096143551C7F,
    0x07F2096143555C7F,0xFF02276143535C7F,0xFF25276150645C7F,0xFF021D6143535C7F,
    0xFF251D6150645C7F,0xFF02066150535C7F,0xFF25066150655C7F,0x07F9096143555D7F,
    0xFF02276143536E7F,0xFF021D6143536E7F,0xFF261D6150646E7F,0xFF02066150536E7F,
    0xFF25276150785C7F,0xFF251D6150785C7F,0x080009615055457F,0xFF2527615064457F,
    0xFF251D615064457F,0xFF0206615053457F,0xFF2527615078457F,0xFF2506615065457F,
    0xFF251D615078457F,0x080709614355117F,0xFF0227614353457F,0xFF021D614353457F,
    0xFF25276150640D7F,0xFF25276150780D7F,0xFF25066150650D7F,0xFF251D6150780D7F,
    0x080E096143556F7F,0xFF26276150646E7F,0xFF0227614353787F,0xFF26276150786E7F,
    0xFF021D614353787F,0xFF26066150656E7F,0xFF261D6150786E7F,0x081509615055787F,
    0xFF2627615064787F,0xFF261D615064787F,0xFF0206615053787F,0xFF2627615078787F,
    0xFF2606615065787F,0xFF261D615078787F,0x081C1C6243551C1C,0x081D186243551C19,
    0x081E186243555D56,0xFF9B736150536E56,0xFF04736150655C56,0xFF04068842595C56,
    0xFF04068842645C56,0xFF1B068842596E56,0xFFB1068842785C56,0x0825736150551C38,
    0xFF04736150535C56,0xFF01736150534556,0xFF01736150654556,0xFF81736150530D56,
    0xFF81736150650D56,0xFF20736150657837,0x082C0688425B1C57,0xFF1B068842646E56,
    0xFF1D068842597856,0xFF1D068842647856,0xFFB1068842786E56,0xFF23068842590D70,
    0xFF1D068842787856,0x0833736150554837,0xFF04736150535C37,0xFF05736150536E37,
    0xFF04736150655C37,0xFF05736150656E37,0xFF20736150537837,0xFF01736150654537,
    0x083A0688425B1470,0xFF04068842595C70,0xFF23068842594570,0xFF23068842644570,
    0xFF23068842640D70,0xFF24068842784570,0xFF24068842780D70,0x0841736150551C10,
    0xFF01736150534537,0xFF01736150530D37,0xFF1E736150656E0A,0xFF0573615053780A,
    0xFF01736150650D37,0xFF0573615065780A,0x0848746143551C1C,0x0849746143551C19,
    0xFF9B736150656E56,0xFF1D736150537856,0xFF23736150530D70,0xFF1D736150657856,
    0xFF23736150650D70,0xFF0579614353780A,0x0850736150554770,0xFF01736150535C70,
    0xFF17736150536E70,0xFF23736150534570,0xFF01736150655C70,0xFF17736150656E70,
    0xFF23736150654570,0x0857736150551C72,0xFF0273615053457F,0xFFA8736150537870,
    0xFFA573615065457F,0xFF02736150530D7F,0xFFA8736150657870,0xFFA5736150650D7F,
    0x085E796143555D0A,0xFF1E796143535C0A,0xFF1E796150645C0A,0xFF1E796143536E0A,
    0xFF1E796150646E0A,0xFF0C796150785C0A,0xFF1E796150786E0A,0x086579614355110A,
    0xFF0C79614353450A,0xFF0C79615064450A,0xFF8C796143530D0A,0xFF8C796150640D0A,
    0xFF0C79615078450A,0xFF8C796150780D0A,0x086C736150555E7F,0xFF02736150535C7F,
    0xFF02736150536E7F,0xFF25736150655C7F,0xFF26736150656E7F,0xFF0273615053787F,
    0xFF2673615065787F,0x08737A6143551C1C,0x0874796143541C38,0xFF04796143535C56,
    0xFF01796143534556,0xFF05796143536E37,0xFF05796150646E37,0xFF81796143530D56,
    0xFF20796143537837,0x087B796143531C57,0xFF9B796143536E56,0xFF01796143535C70,
    0xFF17796143536E70,0xFF23796143534570,0xFF1D796143537856,0xFF23796143530D70,
    0x08827A6143554637,0xFF04796143535C37,0xFF04796150645C37,0xFF01796150644537,
    0xFF01806143534537,0xFFB1796150785C37,0xFFB1796150784537,0x08897A6143551C10,
    0xFF01796143534537,0xFF01796143530D37,0xFF01796150640D37,0xFF01806143530D37,
    0xFFB9796150780D37,0xFF1E79615078780A,0x0890796143531C72,0xFF02796143535C7F,
    0xFF02796143536E7F,0xFF0279614353457F,0xFFA8796143537870,0xFF0279614353787F,
    0xFF02796143530D7F,0x08977A6143541C0A,0xFF1E806143535C0A,0xFF1E806143536E0A,
    0xFF0C80614353450A,0xFF0579615064780A,0xFF0580614353780A,0xFF8C806143530D0A,
    0x089E816143531C1C,0x089F816143531C38,0xFF04806143535C56,0xFF01806143534556,
    0xFF01846143534556,0xFF81806143530D56,0xFF81846143530D56,0xFF20846143537837,
    0x08A6816143531C57,0xFF04846143535C56,0xFF9B806143536E56,0xFF9B846143536E56,
    0xFF1D806143537856,0xFF1D846143537856,0xFF23806143530D70,0x08AD816143534837,
    0xFF04806143535C37,0xFF04846143535C37,0xFF05806143536E37,0xFF05846143536E37,
    0xFF01846143534537,0xFF20806143537837,0x08B4816143531770,0xFF01806143535C70,
    0xFF01846143535C70,0xFF17806143536E70,0xFF23806143534570,0xFF23846143534570,
    0xFF23846143530D70,0x08BB846143531C10,0xFF1E846143535C0A,0xFF1E846143536E0A,
    0xFF0C84614353450A,0xFF01846143530D37,0xFF0584614353780A,0xFF8C846143530D0A,
    0x08C2806143531C72,0xFF02806143535C7F,0xFF02806143536E7F,0xFF0280614353457F,
    0xFFA8806143537870,0xFF0280614353787F,0xFF02806143530D7F,0x08C9856143551C72,
    0x08CA856143555D72,0xFF17846143536E70,0xFF17876142596E70,0xFF02846143536E7F,
    0xFF02876142596E7F,0xFF04876142785C70,0xFF25876142785C7F,0x08D1856143554672,
    0xFF01876142595C70,0xFF01876142645C70,0xFF02846143535C7F,0xFF02876142595C7F,
    0xFF25876142645C7F,0xFFA587614278457F,0x08D8856143554572,0xFF23876142594570,
    0xFF23876142644570,0xFF0284614353457F,0xFF0287614259457F,0xFFA587614264457F,
    0xFF24876142784570,0x08DF856143556F72,0xFF17876142646E70,0xFFA8846143537870,
    0xFF26876142646E7F,0xFF04876142786E70,0xFF0284614353787F,0xFF26876142786E7F,
    0x08E68761425B7872,0xFFA8876142597870,0xFFA8876142647870,0xFF0287614259787F,
    0xFF2687614264787F,0xFF15876142787870,0xFF2687614278787F,0x08ED856143550D72,
    0xFF23876142640D70,0xFF02846143530D7F,0xFF02876142590D7F,0xFFA5876142640D7F,
    0xFF24876142780D70,0xFFA5876142780D7F,0x08F40688425B1C15,0x08F50688425B5C10,
    0xFF04068842595C37,0xFF04068842645C37,0xFF0C068842595C0A,0xFF0C068842645C0A,
    0xFF04068842785C37,0xFF0C068842785C0A,0x08FC0688425B6E10,0xFF05068842596E37,
    0xFF05068842646E37,0xFF1E068842596E0A,0xFF1E068842646E0A,0xFF05068842786E37,
    0xFF1E068842786E0A,0x0903068842654515,0xFF01068842644556,0xFF01068842644537,
    0xFFB1068842784556,0xFF0C06884264450A,0xFFB1068842784537,0xFF0C06884278450A,
    0x090A0688425B1115,0xFF01068842594556,0xFF01068842594537,0xFF0C06884259450A,
    0xFFB1068842780D56,0xFFB1068842780D37,0xFF0C068842780D0A,0x09110688425B7810,
    0xFFA0068842597837,0xFFA0068842647837,0xFF0506884259780A,0xFF0506884264780A,
    0xFFA0068842787837,0xFF1E06884278780A,0x09180688425A0D15,0xFF84068842590D56,
    0xFF84068842640D56,0xFF01068842590D37,0xFF01068842640D37,0xFF0C068842590D0A,
    0xFF0C068842640D0A,0x091F3F8843551C1C,0x09204C88435B1C38,0x09214C88435B5D56,
    0xFF1B498843596E56,0xFF9B668843596E56,0xFF04498843655C56,0xFF04668843655C56,
    0xFF9B738842596E56,0xFF04738842785C56,0x09284C88435B4656,0xFF04498843595C56,
    0xFF04668843595C56,0xFF04738842595C56,0xFF04738842645C56,0xFF01668843654556,
    0xFFB9738842784556,0x092F4C88435B1156,0xFF01498843594556,0xFF01668843594556,
    0xFF01738842594556,0xFF01738842644556,0xFF01498843654556,0xFFB9738842780D56,
    0x09364C88435B6F38,0xFF9B738842646E56,0xFF1B498843656E56,0xFF9B668843656E56,
    0xFF1D738842597856,0xFF1B738842786E56,0xFF20738842597837,0x093D4C88435B7837,
    0xFFA0498843597837,0xFF20668843597837,0xFF20738842647837,0xFFA0498843657837,
    0xFF20668843657837,0xFF20738842787837,0x09444C88435B0D56,0xFF84498843590D56,
    0xFF81668843590D56,0xFF81738842590D56,0xFF81738842640D56,0xFF84498843650D56,
    0xFF81668843650D56,0x094B3E88435B1C57,0x094C3E88435B4670,0xFF04498843595C70,
    0xFF01668843595C70,0xFF043B8842595C70,0xFF01738842595C70,0xFF243B8842784570,
    0xFF24738842784570,0x09533E88435B4570,0xFF23498843594570,0xFF23668843594570,
    0xFF233B8842644570,0xFF23738842644570,0xFF23498843654570,0xFF23668843654570,
    0x095A3E88435B5E57,0xFF1D498843597856,0xFF1D668843597856,0xFF043B8842645C70,
    0xFF01738842645C70,0xFF04498843655C70,0xFF01668843655C70,0x09613E88435B1170,
    0xFF233B8842594570,0xFF23738842594570,0xFF23498843650D70,0xFF23668843650D70,
    0xFF243B8842780D70,0xFF24738842780D70,0x09683E8843657856,0xFF1D3B8842647856,
    0xFF1D738842647856,0xFF1D498843657856,0xFF1D668843657856,0xFF1D3B8842787856,
    0xFF1D738842787856,0x096F3E88435A0D70,0xFF23498843590D70,0xFF23668843590D70,
    0xFF233B8842590D70,0xFF23738842590D70,0xFF233B8842640D70,0xFF23738842640D70,
    0x09764D8843551C10,0x09774C88435B5D37,0xFF05498843596E37,0xFF05668843596E37,
    0xFF04498843655C37,0xFF04668843655C37,0xFF05738842596E37,0xFFB1738842785C37,
    0x097E4C88435B4637,0xFF04498843595C37,0xFF04668843595C37,0xFF04738842595C37,
    0xFF04738842645C37,0xFF01668843654537,0xFFB1738842784537,0x09854D8843556F10,
    0xFF05738842646E37,0xFF05498843656E37,0xFF05668843656E37,0xFFB1738842786E37,
    0xFF0573884259780A,0xFF0579884353780A,0x098C4C88435B1137,0xFF01498843594537,
    0xFF01668843594537,0xFF01738842594537,0xFF01738842644537,0xFF01498843654537,
    0xFFB9738842780D37,0x09934C88435B780A,0xFF0549884359780A,0xFF0566884359780A,
    0xFF0573884264780A,0xFF0549884365780A,0xFF0566884365780A,0xFF1E73884278780A,
    0x099A4C88435B0D37,0xFF01498843590D37,0xFF01668843590D37,0xFF01738842590D37,
    0xFF01738842640D37,0xFF01498843650D37,0xFF01668843650D37,0x09A13E88435B1C72,
    0x09A23E88435B4772,0xFF153B8842596E70,0xFF17738842596E70,0xFF0249884359457F,
    0xFF0266884359457F,0xFF043B8842785C70,0xFF04738842785C70,0x09A93E88435B6E70,
    0xFF15498843596E70,0xFF17668843596E70,0xFF153B8842646E70,0xFF17738842646E70,
    0xFF15498843656E70,0xFF17668843656E70,0x09B03E88435B6F70,0xFF15498843597870,
    0xFFA8668843597870,0xFF153B8842597870,0xFFA8738842597870,0xFF043B8842786E70,
    0xFF04738842786E70,0x09B73E88435B117F,0xFF023B884259457F,0xFF0273884259457F,
    0xFF25498843650D7F,0xFFA5668843650D7F,0xFF253B8842780D7F,0xFFA5738842780D7F,
    0x09BE3E8843657870,0xFF153B8842647870,0xFFA8738842647870,0xFF15498843657870,
    0xFFA8668843657870,0xFF153B8842787870,0xFF15738842787870,0x09C53E88435A0D7F,
    0xFF02498843590D7F,0xFF02668843590D7F,0xFF023B8842590D7F,0xFF02738842590D7F,
    0xFF253B8842640D7F,0xFFA5738842640D7F,0x09CC4D884355170A,0x09CD4C884359170A,
    0xFF0C498843595C0A,0xFF1E498843596E0A,0xFF0C49884359450A,0xFF1E738842595C0A,
    0xFF1E738842596E0A,0xFF0C498843590D0A,0x09D474884353170A,0xFF1E798843535C0A,
    0xFF0C73884259450A,0xFF1E798843536E0A,0xFF0C79884353450A,0xFF8C738842590D0A,
    0xFF8C798843530D0A,0x09DB6788435A170A,0xFF1E668843595C0A,0xFF1E668843596E0A,
    0xFF0C66884359450A,0xFF0C73884264450A,0xFF8C668843590D0A,0xFF8C738842640D0A,
    0x09E24B884365170A,0xFF0C498843655C0A,0xFF1E498843656E0A,0xFF0C49884365450A,
    0xFF0C66884365450A,0xFF0C498843650D0A,0xFF8C668843650D0A,0x09E967884365170A,
    0xFF1E668843655C0A,0xFF1E668843656E0A,0xFF0C738842785C0A,0xFF1E738842786E0A,
    0xFF0C73884278450A,0xFF8C738842780D0A,0x09F074884265170A,0xFF1E738842645C0A,
    0xFF1E738842646E0A,0xFF0C798842655C0A,0xFF1E798842656E0A,0xFF0C79884265450A,
    0xFF8C798842650D0A,0x09F73E88435B487F,0x09F83E88435A5C7F,0xFF02498843595C7F,
    0xFF02668843595C7F,0xFF023B8842595C7F,0xFF02738842595C7F,0xFF253B8842645C7F,
    0xFF25738842645C7F,0x09FF3E88435B5D7F,0xFF25498843655C7F,0xFF25668843655C7F,
    0xFF023B8842596E7F,0xFF02738842596E7F,0xFF253B8842785C7F,0xFF25738842785C7F,
    0x0A063E88435B6E7F,0xFF02498843596E7F,0xFF02668843596E7F,0xFF263B8842646E7F,
    0xFF26738842646E7F,0xFF26498843656E7F,0xFF26668843656E7F,0x0A0D3E884365457F,
    0xFF253B884264457F,0xFFA573884264457F,0xFF2549884365457F,0xFFA566884365457F,
    0xFF253B884278457F,0xFFA573884278457F,0x0A143E88435B6F7F,0xFF0249884359787F,
    0xFF0266884359787F,0xFF023B884259787F,0xFF0273884259787F,0xFF263B8842786E7F,
    0xFF26738842786E7F,0x0A1B3E884365787F,0xFF263B884264787F,0xFF2673884264787F,
    0xFF2649884365787F,0xFF2666884365787F,0xFF263B884278787F,0xFF2673884278787F,
    0x0A220E8843551C1C,0x0A23208843551C38,0x0A242088425B4556,0xFF013B8842594556,
    0xFF013B8842644556,0xFFB13B8842784556,0xFFB12E8842654556,0xFFB1278842654556,
    0xFFB11D8842654556,0x0A2B208843551156,0xFF012E8843534556,0xFF01278843534556,
    0xFF011D8843534556,0xFFB13B8842780D56,0xFFB12E8842650D56,0xFFB1278842650D56,
    0x0A32208843550D56,0xFF843B8842590D56,0xFF842E8843530D56,0xFF843B8842640D56,
    0xFF84278843530D56,0xFF841D8843530D56,0xFFB11D8842650D56,0x0A39208843556E37,
    0xFF053B8842596E37,0xFF052E8843536E37,0xFF053B8842646E37,0xFF05278843536E37,
    0xFF051D8843536E37,0xFF051D8842656E37,0x0A40208843556F37,0xFFA02E8843537837,
    0xFF053B8842786E37,0xFF052E8842656E37,0xFFA0278843537837,0xFF05278842656E37,
    0xFFA01D8843537837,0x0A472088425B7837,0xFFA03B8842597837,0xFFA03B8842647837,
    0xFFA03B8842787837,0xFFA02E8842657837,0xFFA0278842657837,0xFFA01D8842657837,
    0x0A4E208843551C57,0x0A4F208843555C56,0xFF043B8842595C56,0xFF042E8843535C56,
    0xFF043B8842645C56,0xFF04278843535C56,0xFFB1278842655C56,0xFFB11D8842655C56,
    0x0A56208843555D56,0xFF1B3B8842596E56,0xFF1B2E8843536E56,0xFF1B278843536E56,
    0xFFB13B8842785C56,0xFF1B1D8843536E56,0xFFB12E8842655C56,0x0A5D1F8843554657,
    0xFF041D8843535C56,0xFF232E8843534570,0xFF23278843534570,0xFF231D8843534570,
    0xFF24278842654570,0xFF241D8842654570,0x0A64208843556F56,0xFF1B3B8842646E56,
    0xFFB13B8842786E56,0xFFB12E8842656E56,0xFFB1278842656E56,0xFF1D1D8843537856,
    0xFFB11D8842656E56,0x0A6B208843557856,0xFF1D3B8842597856,0xFF1D2E8843537856,
    0xFF1D278843537856,0xFF1D2E8842657856,0xFF1D278842657856,0xFF1D1D8842657856,
    0x0A721F8843550D70,0xFF232E8843530D70,0xFF23278843530D70,0xFF231D8843530D70,
    0xFF242E8842650D70,0xFF24278842650D70,0xFF241D8842650D70,0x0A79208843551C10,
    0x0A7A208843555C37,0xFF043B8842595C37,0xFF042E8843535C37,0xFF043B8842645C37,
    0xFF04278843535C37,0xFF041D8843535C37,0xFF041D8842655C37,0x0A812088425B4537,
    0xFF013B8842594537,0xFF013B8842644537,0xFFB13B8842784537,0xFFB12E8842654537,
    0xFFB1278842654537,0xFFB11D8842654537,0x0A88208843555E10,0xFF043B8842785C37,
    0xFF042E8842655C37,0xFF04278842655C37,0xFF052E884353780A,0xFF0527884353780A,
    0xFF051D884353780A,0x0A8F208843551137,0xFF012E8843534537,0xFF01278843534537,
    0xFF011D8843534537,0xFFB13B8842780D37,0xFFB12E8842650D37,0xFFB1278842650D37,
    0x0A962088425B780A,0xFF053B884259780A,0xFF053B884264780A,0xFF1E3B884278780A,
    0xFF1E2E884265780A,0xFF1E27884265780A,0xFF1E1D884265780A,0x0A9D208843550D37,
    0xFF013B8842590D37,0xFF012E8843530D37,0xFF013B8842640D37,0xFF01278843530D37,
    0xFF011D8843530D37,0xFFB11D8842650D37,0x0AA40B8843551C72,0x0AA50B8843555C70,
    0xFF042E8843535C70,0xFF042E8842655C70,0xFF04068842645C70,0xFF04278842655C70,
    0xFF041D8842655C70,0xFF04068842785C70,0x0AAC0B8843556E70,0xFF152E8843536E70,
    0xFF15278843536E70,0xFF151D8843536E70,0xFF15068842596E70,0xFF15068842646E70,
    0xFF041D8842656E70,0x0AB30B8843551472,0xFF04278843535C70,0xFF041D8843535C70,
    0xFF242E8842654570,0xFF252E8842650D7F,0xFF25278842650D7F,0xFF25068842780D7F,
    0x0ABA0B8843556F70,0xFF152E8843537870,0xFF042E8842656E70,0xFF15278843537870,
    0xFF04278842656E70,0xFF151D8843537870,0xFF04068842786E70,0x0AC10B88425B7870,
    0xFF152E8842657870,0xFF15068842597870,0xFF15068842647870,0xFF15278842657870,
    0xFF151D8842657870,0xFF15068842787870,0x0AC80B8843550D7F,0xFF022E8843530D7F,
    0xFF02278843530D7F,0xFF021D8843530D7F,0xFF02068842590D7F,0xFF25068842640D7F,
    0xFF251D8842650D7F,0x0ACF20884355170A,0x0AD028884353170A,0xFF0C2E8843535C0A,
    0xFF0C278843535C0A,0xFF1E2E8843536E0A,0xFF0C2E884353450A,0xFF1E278843536E0A,
    0xFF0C2E8843530D0A,0x0AD71E884353170A,0xFF0C1D8843535C0A,0xFF0C27884353450A,
    0xFF1E1D8843536E0A,0xFF0C1D884353450A,0xFF0C278843530D0A,0xFF0C1D8843530D0A,
    0x0ADE3B88425B5D0A,0xFF0C3B8842595C0A,0xFF0C3B8842645C0A,0xFF1E3B8842596E0A,
    0xFF1E3B8842646E0A,0xFF0C3B8842785C0A,0xFF1E3B8842786E0A,0x0AE53B88425B110A,
    0xFF0C3B884259450A,0xFF0C3B884264450A,0xFF0C3B8842590D0A,0xFF0C3B8842640D0A,
    0xFF0C3B884278450A,0xFF0C3B8842780D0A,0x0AEC28884265170A,0xFF0C2E8842655C0A,
    0xFF0C278842655C0A,0xFF1E2E8842656E0A,0xFF0C2E884265450A,0xFF1E278842656E0A,
    0xFF0C2E8842650D0A,0x0AF31E884265170A,0xFF0C1D8842655C0A,0xFF0C27884265450A,
    0xFF1E1D8842656E0A,0xFF0C1D884265450A,0xFF0C278842650D0A,0xFF0C1D8842650D0A,
    0x0AFA0B884355487F,0x0AFB28884353487F,0xFF022E8843535C7F,0xFF022E8843536E7F,
    0xFF022E884353457F,0xFF02278843536E7F,0xFF022E884353787F,0xFF0227884353787F,
    0x0B021E884353487F,0xFF02278843535C7F,0xFF021D8843535C7F,0xFF0227884353457F,
    0xFF021D8843536E7F,0xFF021D884353457F,0xFF021D884353787F,0x0B0928884265487F,
    0xFF252E8842655C7F,0xFF262E8842656E7F,0xFF252E884265457F,0xFF26278842656E7F,
    0xFF262E884265787F,0xFF2627884265787F,0x0B101E884265487F,0xFF25278842655C7F,
    0xFF251D8842655C7F,0xFF2527884265457F,0xFF261D8842656E7F,0xFF251D884265457F,
    0xFF261D884265787F,0x0B170688425B467F,0xFF02068842595C7F,0xFF25068842645C7F,
    0xFF0206884259457F,0xFF2506884264457F,0xFF25068842785C7F,0xFF2506884278457F,
    0x0B1E0688425B6F7F,0xFF02068842596E7F,0xFF26068842646E7F,0xFF0206884259787F,
    0xFF2606884264787F,0xFF26068842786E7F,0xFF2606884278787F,0x0B257C8843551C1C,
    0x0B267C8843551C38,0x0B277C8843555C56,0xFF04798843535C56,0xFF04808843535C56,
    0xFF04848843535C56,0xFF04798842655C56,0xFF04878842595C56,0xFF04878842645C56,
    0x0B2E7C88425B4556,0xFFB9798842654556,0xFF01878842594556,0xFF01878842644556,
    0xFFB9808842654556,0xFFB9848842654556,0xFFB9878842784556,0x0B357C8843555E38,
    0xFF04808842655C56,0xFF20798843537837,0xFF04848842655C56,0xFF20808843537837,
    0xFF04878842785C56,0xFF20848843537837,0x0B3C7C8843551156,0xFF01798843534556,
    0xFF01808843534556,0xFF01848843534556,0xFFB9808842650D56,0xFFB9848842650D56,
    0xFFB9878842780D56,0x0B437C88425B7837,0xFF20798842657837,0xFF20878842597837,
    0xFF20878842647837,0xFF20808842657837,0xFF20848842657837,0xFF20878842787837,
    0x0B4A7C8843550D56,0xFF81798843530D56,0xFF81808843530D56,0xFF81848843530D56,
    0xFFB9798842650D56,0xFF81878842590D56,0xFF81878842640D56,0x0B517C8843551C57,
    0x0B527C8843556E56,0xFF9B798843536E56,0xFF9B808843536E56,0xFF9B848843536E56,
    0xFF1B798842656E56,0xFF9B878842596E56,0xFF9B878842646E56,0x0B597C8843556F56,
    0xFF1D798843537856,0xFF1D808843537856,0xFF1B808842656E56,0xFF1D848843537856,
    0xFF1B848842656E56,0xFF1B878842786E56,0x0B607C88425B7856,0xFF1D798842657856,
    0xFF1D878842597856,0xFF1D878842647856,0xFF1D808842657856,0xFF1D848842657856,
    0xFF1D878842787856,0x0B677C88425B4570,0xFF24798842654570,0xFF23878842594570,
    0xFF23878842644570,0xFF24808842654570,0xFF24848842654570,0xFF24878842784570,
    0x0B6E7C8843551170,0xFF23798843534570,0xFF23808843534570,0xFF23848843534570,
    0xFF24808842650D70,0xFF24848842650D70,0xFF24878842780D70,0x0B757C8843550D70,
    0xFF23798843530D70,0xFF23808843530D70,0xFF23848843530D70,0xFF24798842650D70,
    0xFF23878842590D70,0xFF23878842640D70,0x0B7C7C8843551737,0x0B7D7A8843531737,
    0xFF04798843535C37,0xFF05798843536E37,0xFF01798843534537,0xFF01808843534537,
    0xFF01798843530D37,0xFF01808843530D37,0x0B84818843531737,0xFF04808843535C37,
    0xFF04848843535C37,0xFF05808843536E37,0xFF05848843536E37,0xFF01848843534537,
    0xFF01848843530D37,0x0B8B7A8842651737,0xFFB1798842655C37,0xFFB1798842656E37,
    0xFFB1798842654537,0xFFB1808842654537,0xFFB9798842650D37,0xFFB9808842650D37,
    0x0B92818842651737,0xFFB1808842655C37,0xFFB1848842655C37,0xFFB1808842656E37,
    0xFFB1848842656E37,0xFFB1848842654537,0xFFB9848842650D37,0x0B998788425B5D37,
    0xFF04878842595C37,0xFF04878842645C37,0xFF05878842596E37,0xFF05878842646E37,
    0xFFB1878842785C37,0xFFB1878842786E37,0x0BA08788425B1137,0xFF01878842594537,
    0xFF01878842644537,0xFF01878842590D37,0xFF01878842640D37,0xFFB1878842784537,
    0xFFB9878842780D37,0x0BA77C8843551C72,0x0BA87C88425B5C70,0xFF04798842655C70,
    0xFF01878842595C70,0xFF01878842645C70,0xFF04808842655C70,0xFF04848842655C70,
    0xFF04878842785C70,0x0BAF7C8843556E70,0xFF17798843536E70,0xFF17808843536E70,
    0xFF17848843536E70,0xFF04798842656E70,0xFF17878842596E70,0xFF17878842646E70,
    0x0BB67C8843551472,0xFF01798843535C70,0xFF01808843535C70,0xFF01848843535C70,
    0xFFA5808842650D7F,0xFFA5848842650D7F,0xFFA5878842780D7F,0x0BBD7C8843556F70,
    0xFFA8798843537870,0xFFA8808843537870,0xFF04808842656E70,0xFFA8848843537870,
    0xFF04848842656E70,0xFF04878842786E70,0x0BC47C88425B7870,0xFF15798842657870,
    0xFFA8878842597870,0xFFA8878842647870,0xFF15808842657870,0xFF15848842657870,
    0xFF15878842787870,0x0BCB7C8843550D7F,0xFF02798843530D7F,0xFF02808843530D7F,
    0xFF02848843530D7F,0xFFA5798842650D7F,0xFF02878842590D7F,0xFFA5878842640D7F,
    0x0BD27C8843551C0A,0x0BD3828843555D0A,0xFF1E808843536E0A,0xFF1E848843536E0A,
    0xFF0C808842655C0A,0xFF1E878842596E0A,0xFF0C848842655C0A,0xFF0C878842785C0A,
    0x0BDA82884355460A,0xFF1E808843535C0A,0xFF1E848843535C0A,0xFF1E878842595C0A,
    0xFF1E878842645C0A,0xFF0C84884265450A,0xFF0C87884278450A,0x0BE182884355110A,
    0xFF0C80884353450A,0xFF0C84884353450A,0xFF0C87884259450A,0xFF0C87884264450A,
    0xFF0C80884265450A,0xFF8C878842780D0A,0x0BE8828843556F0A,0xFF0580884353780A,
    0xFF1E878842646E0A,0xFF1E808842656E0A,0xFF0584884353780A,0xFF1E848842656E0A,
    0xFF1E878842786E0A,0x0BEF7C88425B780A,0xFF1E79884265780A,0xFF0587884259780A,
    0xFF0587884264780A,0xFF1E80884265780A,0xFF1E84884265780A,0xFF1E87884278780A,
    0x0BF6828843550D0A,0xFF8C808843530D0A,0xFF8C848843530D0A,0xFF8C878842590D0A,
    0xFF8C878842640D0A,0xFF8C808842650D0A,0xFF8C848842650D0A,0x0BFD7C884355487F,
    0x0BFE7A884353487F,0xFF02798843535C7F,0xFF02808843535C7F,0xFF02798843536E7F,
    0xFF0279884353457F,0xFF0280884353457F,0xFF0279884353787F,0x0C0581884353487F,
    0xFF02848843535C7F,0xFF02808843536E7F,0xFF02848843536E7F,0xFF0284884353457F,
    0xFF0280884353787F,0xFF0284884353787F,0x0C0C7A884265487F,0xFF25798842655C7F,
    0xFF25808842655C7F,0xFF26798842656E7F,0xFFA579884265457F,0xFFA580884265457F,
    0xFF2679884265787F,0x0C1381884265487F,0xFF25848842655C7F,0xFF26808842656E7F,
    0xFF26848842656E7F,0xFFA584884265457F,0xFF2680884265787F,0xFF2684884265787F,
    0x0C1A8788425B467F,0xFF02878842595C7F,0xFF25878842645C7F,0xFF0287884259457F,
    0xFFA587884264457F,0xFF25878842785C7F,0xFFA587884278457F,0x0C218788425B6F7F,
    0xFF02878842596E7F,0xFF26878842646E7F,0xFF0287884259787F,0xFF2687884264787F,
    0xFF26878842786E7F,0xFF2687884278787F,0x0C281C6269551C1C,0x0C2932617D551C1C,
    0x0C2A3D617D551C38,0x0C2B3D617D5B4556,0xFF0149617D594556,0xFFA366617D594556,
    0xFF013B617D594556,0xFF0149617D654556,0xFF0166617D654556,0xFF033B617D654556,
    0x0C323D617D551156,0xFF1049617D524556,0xFFA366617D524556,0xFF103B617D524556,
    0xFF8449617D650D56,0xFF8166617D650D56,0xFF033B617D650D56,0x0C393D617D530D56,
    0xFF8449617D590D56,0xFFA366617D590D56,0xFF1049617D520D56,0xFFA366617D520D56,
    0xFF843B617D590D56,0xFF103B617D520D56,0x0C403D617D536E37,0xFF0549617D596E37,
    0xFF0566617D596E37,0xFF1049617D526E37,0xFF0566617D526E37,0xFF053B617D596E37,
    0xFF103B617D526E37,0x0C473D617D556F37,0xFF0549617D656E37,0xFF0566617D656E37,
    0xFF1049617D527837,0xFF2066617D527837,0xFF223B617D656E37,0xFF103B617D527837,
    0x0C4E3D617D5B7837,0xFFA049617D597837,0xFF2066617D597837,0xFFA03B617D597837,
    0xFFA049617D657837,0xFF2066617D657837,0xFF223B617D657837,0x0C553D617D551C57,
    0x0C563D617D5B5C56,0xFF0449617D595C56,0xFFA366617D595C56,0xFF043B617D595C56,
    0xFF0449617D655C56,0xFF0466617D655C56,0xFF033B617D655C56,0x0C5D3D617D536E56,
    0xFF1B49617D596E56,0xFFA366617D596E56,0xFF1049617D526E56,0xFFA366617D526E56,
    0xFF1B3B617D596E56,0xFF103B617D526E56,0x0C643D617D551457,0xFF1049617D525C56,
    0xFFA366617D525C56,0xFF103B617D525C56,0xFF2349617D650D70,0xFF2366617D650D70,
    0xFF233B617D650D70,0x0C6B3D617D556F56,0xFF1B49617D656E56,0xFF9B66617D656E56,
    0xFF1049617D527856,0xFF1D66617D527856,0xFF223B617D656E56,0xFF103B617D527856,
    0x0C723D617D5B7856,0xFF1D49617D597856,0xFF1D66617D597856,0xFF1D3B617D597856,
    0xFF1D49617D657856,0xFF1D66617D657856,0xFF223B617D657856,0x0C793D617D530D70,
    0xFF2349617D590D70,0xFFA366617D590D70,0xFF1049617D520D70,0xFFA366617D520D70,
    0xFF233B617D590D70,0xFF103B617D520D70,0x0C803D617D551C10,0x0C813D617D535C37,
    0xFF0449617D595C37,0xFF0466617D595C37,0xFF1049617D525C37,0xFF0466617D525C37,
    0xFF043B617D595C37,0xFF103B617D525C37,0x0C883D617D5B4537,0xFF0149617D594537,
    0xFF0166617D594537,0xFF013B617D594537,0xFF0149617D654537,0xFF0166617D654537,
    0xFF833B617D654537,0x0C8F3D617D555E10,0xFF0449617D655C37,0xFF0466617D655C37,
    0xFF033B617D655C37,0xFF1A49617D52780A,0xFF0566617D52780A,0xFF1A3B617D52780A,
    0x0C963D617D551137,0xFF1049617D524537,0xFF0166617D524537,0xFF103B617D524537,
    0xFF0149617D650D37,0xFF0166617D650D37,0xFF833B617D650D37,0x0C9D3D617D5B780A,
    0xFF0549617D59780A,0xFF0566617D59780A,0xFF053B617D59780A,0xFF0549617D65780A,
    0xFF0566617D65780A,0xFF223B617D65780A,0x0CA43D617D530D37,0xFF0149617D590D37,
    0xFF0166617D590D37,0xFF1049617D520D37,0xFF0166617D520D37,0xFF013B617D590D37,
    0xFF103B617D520D37,0x0CAB3D617D551C72,0x0CAC3D617D535C70,0xFF0449617D595C70,
    0xFFA366617D595C70,0xFF1049617D525C70,0xFFA366617D525C70,0xFF043B617D595C70,
    0xFF103B617D525C70,0x0CB33D617D555D70,0xFF1049617D526E70,0xFFA366617D526E70,
    0xFF0449617D655C70,0xFF0166617D655C70,0xFF103B617D526E70,0xFF043B617D655C70,
    0x0CBA3D617D5B6E70,0xFF1549617D596E70,0xFFA366617D596E70,0xFF153B617D596E70,
    0xFF1549617D656E70,0xFF1766617D656E70,0xFF153B617D656E70,0x0CC13D617D5B4570,
    0xFF2349617D594570,0xFFA366617D594570,0xFF233B617D594570,0xFF2349617D654570,
    0xFF2366617D654570,0xFF233B617D654570,0x0CC83D617D531172,0xFF1049617D524570,
    0xFFA366617D524570,0xFF103B617D524570,0xFF0249617D520D7F,0xFF023B617D590D7F,
    0xFF023B617D520D7F,0x0CCF3C617D557870,0xFF1549617D597870,0xFF1049617D527870,
    0xFF153B617D597870,0xFF103B617D527870,0xFF1549617D657870,0xFF153B617D657870,
    0x0CD63D617D55170A,0x0CD73D617D535C0A,0xFF0C49617D595C0A,0xFF1E66617D595C0A,
    0xFF1A49617D525C0A,0xFF1E66617D525C0A,0xFF0C3B617D595C0A,0xFF1A3B617D525C0A,
    0x0CDE3D617D555D0A,0xFF1A49617D526E0A,0xFF1E66617D526E0A,0xFF0C49617D655C0A,
    0xFF1E66617D655C0A,0xFF1A3B617D526E0A,0xFF033B617D655C0A,0x0CE53D617D5B6E0A,
    0xFF1E49617D596E0A,0xFF1E66617D596E0A,0xFF1E3B617D596E0A,0xFF1E49617D656E0A,
    0xFF1E66617D656E0A,0xFF223B617D656E0A,0x0CEC3D617D5B450A,0xFF0C49617D59450A,
    0xFF0C66617D59450A,0xFF0C3B617D59450A,0xFF0C49617D65450A,0xFF0C66617D65450A,
    0xFF833B617D65450A,0x0CF33D617D55110A,0xFF1A49617D52450A,0xFF0C66617D52450A,
    0xFF1A3B617D52450A,0xFF0C49617D650D0A,0xFF8C66617D650D0A,0xFF833B617D650D0A,
    0x0CFA3D617D530D0A,0xFF0C49617D590D0A,0xFF8C66617D590D0A,0xFF1A49617D520D0A,
    0xFF8C66617D520D0A,0xFF0C3B617D590D0A,0xFF1A3B617D520D0A,0x0D0130617D551C7F,
    0x0D0230617D555C7F,0xFF0249617D595C7F,0xFF0249617D525C7F,0xFF023B617D595C7F,
    0xFF023B617D525C7F,0xFF022E617D535C7F,0xFF022E617D655C7F,0x0D0930617D555D7F,
    0xFF0249617D526E7F,0xFF023B617D596E7F,0xFF0249617D655C7F,0xFF023B617D526E7F,
    0xFF022E617D536E7F,0xFF023B617D655C7F,0x0D1030617D55457F,0xFF0249617D59457F,
    0xFF023B617D59457F,0xFF022E617D53457F,0xFF0249617D65457F,0xFF023B617D65457F,
    0xFF022E617D65457F,0x0D1730617D55117F,0xFF0249617D52457F,0xFF023B617D52457F,
    0xFF0249617D590D7F,0xFF0249617D650D7F,0xFF023B617D650D7F,0xFF022E617D650D7F,
    0x0D1E30617D556F7F,0xFF0249617D596E7F,0xFF0249617D656E7F,0xFF0249617D52787F,
    0xFF023B617D656E7F,0xFF023B617D52787F,0xFF022E617D656E7F,0x0D2530617D55787F,
    0xFF0249617D59787F,0xFF023B617D59787F,0xFF022E617D53787F,0xFF0249617D65787F,
    0xFF023B617D65787F,0xFF022E617D65787F,0x0D2C6C617D551C1C,0x0D2D76617D551C38,
    0x0D2E76617D554656,0xFFA373617D525C56,0xFFA373617D594556,0xFF0373617D654556,
    0xFFA384617D534556,0xFF0379617D654556,0xFF0380617D654556,0x0D3576617D551156,
    0xFFA373617D524556,0xFFA379617D534556,0xFFA380617D534556,0xFF0379617D650D56,
    0xFF0380617D650D56,0xFF0384617D650D56,0x0D3C76617D550D56,0xFFA373617D590D56,
    0xFFA373617D520D56,0xFFA379617D530D56,0xFFA380617D530D56,0xFF0373617D650D56,
    0xFFA384617D530D56,0x0D4376617D556E37,0xFF0573617D596E37,0xFF0573617D526E37,
    0xFF0579617D536E37,0xFF2280617D536E37,0xFF2273617D656E37,0xFF0584617D536E37,
    0x0D4A76617D556F37,0xFF2073617D527837,0xFF2279617D656E37,0xFF2079617D537837,
    0xFF2280617D656E37,0xFF2280617D537837,0xFF2284617D656E37,0x0D5176617D557837,
    0xFF2073617D597837,0xFF2273617D657837,0xFF2084617D537837,0xFF2279617D657837,
    0xFF2280617D657837,0xFF2284617D657837,0x0D5875617D551C57,0x0D5975617D555D56,
    0xFFA373617D595C56,0xFFA373617D526E56,0xFFA379617D536E56,0xFF0373617D655C56,
    0xFF0379617D655C56,0xFF0380617D655C56,0x0D6075617D554657,0xFFA379617D535C56,
    0xFFA380617D535C56,0xFFA373617D525C70,0xFFA379617D535C70,0xFF2379617D654570,
    0xFF2380617D654570,0x0D6775617D556F56,0xFFA373617D596E56,0xFFA380617D536E56,
    0xFF2273617D656E56,0xFF1D73617D527856,0xFF2279617D656E56,0xFF2280617D656E56,
    0x0D6E75617D551170,0xFFA373617D594570,0xFFA373617D524570,0xFFA379617D534570,
    0xFFA380617D534570,0xFF2373617D654570,0xFF2380617D650D70,0x0D7575617D557856,
    0xFF1D73617D597856,0xFF1D79617D537856,0xFF2280617D537856,0xFF2273617D657856,
    0xFF2279617D657856,0xFF2280617D657856,0x0D7C75617D550D70,0xFFA373617D590D70,
    0xFFA373617D520D70,0xFFA379617D530D70,0xFFA380617D530D70,0xFF2373617D650D70,
    0xFF2379617D650D70,0x0D8376617D551C10,0x0D8476617D555C37,0xFF0473617D595C37,
    0xFF0473617D525C37,0xFF0479617D535C37,0xFF0380617D535C37,0xFF0373617D655C37,
    0xFF0484617D535C37,0x0D8B76617D554537,0xFF0173617D594537,0xFF8373617D654537,
    0xFF0184617D534537,0xFF8379617D654537,0xFF8380617D654537,0xFF8384617D654537,
    0x0D9276617D555E10,0xFF0379617D655C37,0xFF0380617D655C37,0xFF0384617D655C37,
    0xFF0573617D52780A,0xFF0579617D53780A,0xFF2280617D53780A,0x0D9976617D551137,
    0xFF0173617D524537,0xFF0179617D534537,0xFF8380617D534537,0xFF8379617D650D37,
    0xFF8380617D650D37,0xFF8384617D650D37,0x0DA076617D55780A,0xFF0573617D59780A,
    0xFF2273617D65780A,0xFF0584617D53780A,0xFF2279617D65780A,0xFF2280617D65780A,
    0xFF2284617D65780A,0x0DA776617D550D37,0xFF0173617D590D37,0xFF0173617D520D37,
    0xFF0179617D530D37,0xFF8380617D530D37,0xFF8373617D650D37,0xFF0184617D530D37,
    0x0DAE6B617D551C72,0x0DAF75617D555D70,0xFFA373617D596E70,0xFFA373617D526E70,
    0xFFA379617D536E70,0xFFA380617D536E70,0xFF1773617D656E70,0xFF0180617D655C70,
    0x0DB675617D554672,0xFFA373617D595C70,0xFFA380617D535C70,0xFF0173617D655C70,
    0xFF0179617D655C70,0xFFA379617D53457F,0xFFA380617D53457F,0x0DBD6B617D556F70,
    0xFFA866617D527870,0xFFA873617D527870,0xFF1779617D656E70,0xFFA879617D537870,
    0xFF1780617D656E70,0xFFA880617D537870,0x0DC46B617D55117F,0xFFA366617D52457F,
    0xFFA373617D52457F,0xFF0266617D650D7F,0xFF0273617D650D7F,0xFF0279617D650D7F,
    0xFF0280617D650D7F,0x0DCB6B617D5B7870,0xFFA866617D597870,0xFFA873617D597870,
    0xFFA866617D657870,0xFFA873617D657870,0xFFA879617D657870,0xFFA880617D657870,
    0x0DD26B617D530D7F,0xFFA366617D590D7F,0xFFA366617D520D7F,0xFFA373617D590D7F,
    0xFFA373617D520D7F,0xFFA379617D530D7F,0xFFA380617D530D7F,0x0DD976617D55170A,
    0x0DDA74617D53170A,0xFF1E73617D525C0A,0xFF1E73617D526E0A,0xFF0C73617D52450A,
    0xFF0C79617D53450A,0xFF8C73617D520D0A,0xFF8C79617D530D0A,0x0DE17A617D53170A,
    0xFF1E79617D535C0A,0xFF0380617D535C0A,0xFF1E79617D536E0A,0xFF2280617D536E0A,
    0xFF8380617D53450A,0xFF8380617D530D0A,0x0DE876617D53170A,0xFF0C73617D59450A,
    0xFF1E84617D535C0A,0xFF8C73617D590D0A,0xFF1E84617D536E0A,0xFF0C84617D53450A,
    0xFF8C84617D530D0A,0x0DEF73617D5B170A,0xFF1E73617D595C0A,0xFF1E73617D596E0A,
    0xFF0373617D655C0A,0xFF2273617D656E0A,0xFF8373617D65450A,0xFF8373617D650D0A,
    0x0DF67A617D65170A,0xFF0379617D655C0A,0xFF2279617D656E0A,0xFF8379617D65450A,
    0xFF8380617D65450A,0xFF8379617D650D0A,0xFF8380617D650D0A,0x0DFD81617D65170A,
    0xFF0380617D655C0A,0xFF0384617D655C0A,0xFF2280617D656E0A,0xFF2284617D656E0A,
    0xFF8384617D65450A,0xFF8384617D650D0A,0x0E046B617D55487F,0x0E056B617D535C7F,
    0xFFA366617D595C7F,0xFFA366617D525C7F,0xFFA373617D595C7F,0xFFA373617D525C7F,
    0xFFA379617D535C7F,0xFFA380617D535C7F,0x0E0C6B617D555D7F,0xFFA366617D526E7F,
    0xFF0266617D655C7F,0xFFA373617D526E7F,0xFF0273617D655C7F,0xFF0279617D655C7F,
    0xFF0280617D655C7F,0x0E136B617D556E7F,0xFFA366617D596E7F,0xFFA373617D596E7F,
    0xFFA379617D536E7F,0xFF0266617D656E7F,0xFFA380617D536E7F,0xFF0273617D656E7F,
    0x0E1A6B617D5B457F,0xFFA366617D59457F,0xFFA373617D59457F,0xFF0266617D65457F,
    0xFF0273617D65457F,0xFF0279617D65457F,0xFF0280617D65457F,0x0E216B617D556F7F,
    0xFF0266617D52787F,0xFF0273617D52787F,0xFF0279617D656E7F,0xFF0279617D53787F,
    0xFF0280617D656E7F,0xFF0280617D53787F,0x0E286B617D5B787F,0xFF0266617D59787F,
    0xFF0273617D59787F,0xFF0266617D65787F,0xFF0273617D65787F,0xFF0279617D65787F,
    0xFF0280617D65787F,0x0E2F0B617D551C1C,0x0E300B617D551C38,0x0E310B617D554556,
    0xFF012E617D534556,0xFF032E617D654556,0xFF0106617D594556,0xFF0327617D654556,
    0xFF031D617D654556,0xFF0306617D654556,0x0E380B617D551156,0xFF0327617D534556,
    0xFF011D617D534556,0xFF1006617D524556,0xFF032E617D650D56,0xFF0327617D650D56,
    0xFF031D617D650D56,0x0E3F0B617D550D56,0xFF842E617D530D56,0xFF0327617D530D56,
    0xFF841D617D530D56,0xFF8406617D590D56,0xFF1006617D520D56,0xFF0306617D650D56,
    0x0E460B617D556E37,0xFF052E617D536E37,0xFF2227617D536E37,0xFF051D617D536E37,
    0xFF0506617D596E37,0xFF1006617D526E37,0xFF2206617D656E37,0x0E4D0B617D556F37,
    0xFF222E617D656E37,0xFF2227617D656E37,0xFF2227617D537837,0xFF221D617D656E37,
    0xFFA01D617D537837,0xFF1006617D527837,0x0E540B617D557837,0xFFA02E617D537837,
    0xFF222E617D657837,0xFFA006617D597837,0xFF2227617D657837,0xFF221D617D657837,
    0xFF2206617D657837,0x0E5B0B617D551C57,0x0E5C0B617D555C56,0xFF042E617D535C56,
    0xFF032E617D655C56,0xFF0406617D595C56,0xFF0327617D655C56,0xFF031D617D655C56,
    0xFF0306617D655C56,0x0E630B617D556E56,0xFF1B2E617D536E56,0xFF2227617D536E56,
    0xFF1B1D617D536E56,0xFF1B06617D596E56,0xFF1006617D526E56,0xFF2206617D656E56,
    0x0E6A0B617D551457,0xFF0327617D535C56,0xFF041D617D535C56,0xFF1006617D525C56,
    0xFF232E617D650D70,0xFF2327617D650D70,0xFF231D617D650D70,0x0E710B617D556F56,
    0xFF222E617D656E56,0xFF2227617D656E56,0xFF2227617D537856,0xFF221D617D656E56,
    0xFF1D1D617D537856,0xFF1006617D527856,0x0E780B617D557856,0xFF1D2E617D537856,
    0xFF222E617D657856,0xFF1D06617D597856,0xFF2227617D657856,0xFF221D617D657856,
    0xFF2206617D657856,0x0E7F0B617D550D70,0xFF232E617D530D70,0xFF2327617D530D70,
    0xFF231D617D530D70,0xFF2306617D590D70,0xFF1006617D520D70,0xFF2306617D650D70,
    0x0E860B617D554870,0x0E870B617D555C70,0xFF042E617D535C70,0xFF042E617D655C70,
    0xFF0406617D595C70,0xFF0427617D655C70,0xFF041D617D655C70,0xFF0406617D655C70,
    0x0E8E0B617D554670,0xFF0427617D535C70,0xFF041D617D535C70,0xFF1006617D525C70,
    0xFF232E617D654570,0xFF2327617D654570,0xFF231D617D654570,0x0E950B617D556E70,
    0xFF152E617D536E70,0xFF1527617D536E70,0xFF151D617D536E70,0xFF1506617D596E70,
    0xFF1006617D526E70,0xFF1506617D656E70,0x0E9C0B617D554570,0xFF232E617D534570,
    0xFF2327617D534570,0xFF231D617D534570,0xFF2306617D594570,0xFF1006617D524570,
    0xFF2306617D654570,0x0EA30B617D556F70,0xFF152E617D656E70,0xFF1527617D656E70,
    0xFF1527617D537870,0xFF151D617D656E70,0xFF151D617D537870,0xFF1006617D527870,
    0x0EAA0B617D557870,0xFF152E617D537870,0xFF152E617D657870,0xFF1506617D597870,
    0xFF1527617D657870,0xFF151D617D657870,0xFF1506617D657870,0x0EB10B617D551C10,
    0x0EB20B617D555C37,0xFF042E617D535C37,0xFF0327617D535C37,0xFF041D617D535C37,
    0xFF0406617D595C37,0xFF1006617D525C37,0xFF0306617D655C37,0x0EB90B617D554537,
    0xFF012E617D534537,0xFF832E617D654537,0xFF0106617D594537,0xFF8327617D654537,
    0xFF831D617D654537,0xFF8306617D654537,0x0EC00B617D555E10,0xFF032E617D655C37,
    0xFF0327617D655C37,0xFF031D617D655C37,0xFF2227617D53780A,0xFF051D617D53780A,
    0xFF1A06617D52780A,0x0EC70B617D551137,0xFF8327617D534537,0xFF011D617D534537,
    0xFF1006617D524537,0xFF832E617D650D37,0xFF8327617D650D37,0xFF831D617D650D37,
    0x0ECE0B617D55780A,0xFF052E617D53780A,0xFF222E617D65780A,0xFF0506617D59780A,
    0xFF2227617D65780A,0xFF221D617D65780A,0xFF2206617D65780A,0x0ED50B617D550D37,
    0xFF012E617D530D37,0xFF8327617D530D37,0xFF011D617D530D37,0xFF0106617D590D37,
    0xFF1006617D520D37,0xFF8306617D650D37,0x0EDC0B617D55170A,0x0EDD0B617D555C0A,
    0xFF0C2E617D535C0A,0xFF0327617D535C0A,0xFF0C1D617D535C0A,0xFF0C06617D595C0A,
    0xFF1A06617D525C0A,0xFF0306617D655C0A,0x0EE40B617D555D0A,0xFF2227617D536E0A,
    0xFF032E617D655C0A,0xFF1E1D617D536E0A,0xFF0327617D655C0A,0xFF1A06617D526E0A,
    0xFF031D617D655C0A,0x0EEB0B617D556E0A,0xFF1E2E617D536E0A,0xFF222E617D656E0A,
    0xFF1E06617D596E0A,0xFF2227617D656E0A,0xFF221D617D656E0A,0xFF2206617D656E0A,
    0x0EF20B617D55450A,0xFF0C2E617D53450A,0xFF832E617D65450A,0xFF0C06617D59450A,
    0xFF8327617D65450A,0xFF831D617D65450A,0xFF8306617D65450A,0x0EF90B617D55110A,
    0xFF8327617D53450A,0xFF0C1D617D53450A,0xFF1A06617D52450A,0xFF832E617D650D0A,
    0xFF8327617D650D0A,0xFF831D617D650D0A,0x0F000B617D550D0A,0xFF0C2E617D530D0A,
    0xFF8327617D530D0A,0xFF0C1D617D530D0A,0xFF0C06617D590D0A,0xFF1A06617D520D0A,
    0xFF8306617D650D0A,0x0F070B617D551C7F,0x0F0809617D555D7F,0xFF0206617D595C7F,
    0xFF021D617D536E7F,0xFF0227617D655C7F,0xFF0206617D526E7F,0xFF021D617D655C7F,
    0xFF0206617D655C7F,0x0F0F09617D55467F,0xFF0227617D535C7F,0xFF021D617D535C7F,
    0xFF0206617D525C7F,0xFF0227617D65457F,0xFF021D617D65457F,0xFF0206617D65457F,
    0x0F1609617D55117F,0xFF0227617D53457F,0xFF021D617D53457F,0xFF0206617D59457F,
    0xFF0206617D52457F,0xFF0227617D650D7F,0xFF021D617D650D7F,0x0F1D09617D556F7F,
    0xFF0227617D536E7F,0xFF0206617D596E7F,0xFF0227617D656E7F,0xFF021D617D656E7F,
    0xFF0206617D656E7F,0xFF0206617D52787F,0x0F2409617D55787F,0xFF0227617D53787F,
    0xFF021D617D53787F,0xFF0206617D59787F,0xFF0227617D65787F,0xFF021D617D65787F,
    0xFF0206617D65787F,0x0F2B0B617D550D7F,0xFF022E617D530D7F,0xFF0227617D530D7F,
    0xFF021D617D530D7F,0xFF0206617D590D7F,0xFF0206617D520D7F,0xFF0206617D650D7F,
    0x0F321C627D551C1C,0x0F3324627D551C1C,0x0F3484617D554856,0xFFA384617D535C56,
    0xFFA384617D536E56,0xFF0384617D655C56,0xFF2284617D656E56,0xFF1D84617D537856,
    0xFF0384617D654556,0x0F3B84617D551C57,0xFFA384617D535C70,0xFFA384617D534570,
    0xFF2384617D654570,0xFFA384617D530D70,0xFF2284617D657856,0xFF2384617D650D70,
    0x0F4284617D551C72,0xFFA384617D536E70,0xFF0184617D655C70,0xFF1784617D656E70,
    0xFFA884617D537870,0xFFA384617D530D7F,0xFFA884617D657870,0x0F4984617D55177F,
    0xFFA384617D535C7F,0xFFA384617D536E7F,0xFFA384617D53457F,0xFF0284617D655C7F,
    0xFF0284617D65457F,0xFF0284617D650D7F,0x0F5085617D551C1C,0xFF0284617D656E7F,
    0xFF0284617D53787F,0xFF8C87617D590D0A,0xFF8C87617D520D0A,0xFF0284617D65787F,
    0xFF8387617D650D0A,0x0F571D887D554870,0xFF041D887D535C70,0xFF151D887D536E70,
    0xFF041D887D655C70,0xFF151D887D656E70,0xFF151D887D537870,0xFF231D887D654570,
    0x0F5E08887D551C1C,0x0F5F1D887D555E56,0xFF041D887D535C56,0xFF1B1D887D536E56,
    0xFF031D887D655C56,0xFF221D887D656E56,0xFF1D1D887D537856,0xFF221D887D657856,
    0x0F661D887D551C38,0xFF011D887D534556,0xFF031D887D654556,0xFF841D887D530D56,
    0xFFA01D887D537837,0xFF031D887D650D56,0xFF221D887D657837,0x0F6D1D887D554737,
    0xFF041D887D535C37,0xFF051D887D536E37,0xFF011D887D534537,0xFF031D887D655C37,
    0xFF221D887D656E37,0xFF831D887D654537,0x0F741D887D551C10,0xFF1E1D887D536E0A,
    0xFF011D887D530D37,0xFF221D887D656E0A,0xFF051D887D53780A,0xFF831D887D650D37,
    0xFF221D887D65780A,0x0F7B08887D551C72,0xFF231D887D534570,0xFF231D887D530D70,
    0xFF231D887D650D70,0xFF0206887D59787F,0xFF0206887D52787F,0xFF0206887D65787F,
    0x0F821D887D55140A,0xFF0C1D887D535C0A,0xFF0C1D887D53450A,0xFF031D887D655C0A,
    0xFF831D887D65450A,0xFF0C1D887D530D0A,0xFF831D887D650D0A,0x0F8987617D551C15,
    0x0F8A87617D5B5C15,0xFFA387617D595C56,0xFF0487617D595C37,0xFF0387617D655C56,
    0xFF1E87617D595C0A,0xFF0387617D655C37,0xFF0387617D655C0A,0x0F9187617D554615,
    0xFFA387617D525C56,0xFF0487617D525C37,0xFF0387617D654556,0xFF1E87617D525C0A,
    0xFF8387617D654537,0xFF8387617D65450A,0x0F9887617D556E10,0xFF0587617D596E37,
    0xFF0587617D526E37,0xFF1E87617D596E0A,0xFF2287617D656E37,0xFF1E87617D526E0A,
    0xFF2287617D656E0A,0x0F9F87617D534515,0xFFA387617D594556,0xFFA387617D524556,
    0xFF0187617D594537,0xFF0187617D524537,0xFF0C87617D59450A,0xFF0C87617D52450A,
    0x0FA687617D557810,0xFF2087617D597837,0xFF2087617D527837,0xFF0587617D59780A,
    0xFF2287617D657837,0xFF0587617D52780A,0xFF2287617D65780A,0x0FAD87617D550D38,
    0xFFA387617D590D56,0xFFA387617D520D56,0xFF0187617D590D37,0xFF0187617D520D37,
    0xFF0387617D650D56,0xFF8387617D650D37,0x0FB406887D551C15,0x0FB506887D555C10,
    0xFF0406887D595C37,0xFF1006887D525C37,0xFF0C06887D595C0A,0xFF0306887D655C37,
    0xFF1A06887D525C0A,0xFF0306887D655C0A,0x0FBC06887D556E10,0xFF0506887D596E37,
    0xFF1006887D526E37,0xFF1E06887D596E0A,0xFF2206887D656E37,0xFF1A06887D526E0A,
    0xFF2206887D656E0A,0x0FC306887D5B4515,0xFF0106887D594556,0xFF0106887D594537,
    0xFF0306887D654556,0xFF0C06887D59450A,0xFF8306887D654537,0xFF8306887D65450A,
    0x0FCA06887D551115,0xFF1006887D524556,0xFF1006887D524537,0xFF1A06887D52450A,
    0xFF0306887D650D56,0xFF8306887D650D37,0xFF8306887D650D0A,0x0FD106887D557810,
    0xFFA006887D597837,0xFF1006887D527837,0xFF0506887D59780A,0xFF2206887D657837,
    0xFF1A06887D52780A,0xFF2206887D65780A,0x0FD806887D530D15,0xFF8406887D590D56,
    0xFF1006887D520D56,0xFF0106887D590D37,0xFF1006887D520D37,0xFF0C06887D590D0A,
    0xFF1A06887D520D0A,0x0FDF87617D551C58,0x0FE087617D555C72,0xFFA387617D595C70,
    0xFFA387617D525C70,0xFFA387617D595C7F,0xFF0187617D655C70,0xFFA387617D525C7F,
    0xFF0287617D655C7F,0x0FE787617D536E58,0xFFA387617D596E56,0xFFA387617D526E56,
    0xFFA387617D596E70,0xFFA387617D526E70,0xFFA387617D596E7F,0xFFA387617D526E7F,
    0x0FEE87617D554572,0xFFA387617D594570,0xFFA387617D524570,0xFFA387617D59457F,
    0xFF2387617D654570,0xFFA387617D52457F,0xFF0287617D65457F,0x0FF587617D556F58,
    0xFF2287617D656E56,0xFF1D87617D527856,0xFF1787617D656E70,0xFFA887617D527870,
    0xFF0287617D656E7F,0xFF0287617D52787F,0x0FFC87617D5B7858,0xFF1D87617D597856,
    0xFFA887617D597870,0xFF2287617D657856,0xFF0287617D59787F,0xFFA887617D657870,
    0xFF0287617D65787F,0x100387617D550D72,0xFFA387617D590D70,0xFFA387617D520D70,
    0xFFA387617D590D7F,0xFF2387617D650D70,0xFFA387617D520D7F,0xFF0287617D650D7F,
    0x100A06887D551C58,0x100B06887D535C58,0xFF0406887D595C56,0xFF1006887D525C56,
    0xFF0406887D595C70,0xFF1006887D525C70,0xFF0206887D595C7F,0xFF0206887D525C7F,
    0x101206887D555D58,0xFF1006887D526E56,0xFF0306887D655C56,0xFF1006887D526E70,
    0xFF0406887D655C70,0xFF0206887D526E7F,0xFF0206887D655C7F,0x101906887D5B6E58,
    0xFF1B06887D596E56,0xFF1506887D596E70,0xFF2206887D656E56,0xFF0206887D596E7F,
    0xFF1506887D656E70,0xFF0206887D656E7F,0x102006887D554572,0xFF2306887D594570,
    0xFF1006887D524570,0xFF0206887D59457F,0xFF2306887D654570,0xFF0206887D52457F,
    0xFF0206887D65457F,0x102706887D557857,0xFF1D06887D597856,0xFF1006887D527856,
    0xFF1506887D597870,0xFF1006887D527870,0xFF2206887D657856,0xFF1506887D657870,
    0x102E06887D550D72,0xFF2306887D590D70,0xFF1006887D520D70,0xFF0206887D590D7F,
    0xFF2306887D650D70,0xFF0206887D520D7F,0xFF0206887D650D7F,0x1035418869551C1C,
    0x10364F8869551C38,0x10374F8869555D37,0xFF044988685B5C37,0xFF046688685B5C37,
    0xFF0549887D596E37,0xFF1049887D526E37,0xFF0449887D655C37,0xFF04878868555C37,
    0x103E4F8868556E37,0xFF054988685B6E37,0xFF05738868556E37,0xFF05798868556E37,
    0xFF05808868556E37,0xFF05848868556E37,0xFF05878868556E37,0x10454E8868551438,
    0xFF04738868555C37,0xFF04798868555C37,0xFF844988685B0D56,0xFF816688685B0D56,
    0xFF04808868555C37,0xFF04848868555C37,0x104C4D8869556F37,0xFF056688685B6E37,
    0xFFA049887D597837,0xFF0549887D656E37,0xFF1049887D527837,0xFF20738868557837,
    0xFF20798868557837,0x10534F8869557837,0xFFA04988685B7837,0xFF206688685B7837,
    0xFFA049887D657837,0xFF20808868557837,0xFF20848868557837,0xFF20878868557837,
    0x105A4F8869550D56,0xFF1049887D520D56,0xFF81738868550D56,0xFF81798868550D56,
    0xFF81808868550D56,0xFF81848868550D56,0xFF81878868550D56,0x10614F8868551C57,
    0x10624F8868555D56,0xFF044988685B5C56,0xFF046688685B5C56,0xFF9B738868556E56,
    0xFF9B798868556E56,0xFF04848868555C56,0xFF04878868555C56,0x10694F8868554656,
    0xFF04738868555C56,0xFF014988685B4556,0xFF016688685B4556,0xFF04798868555C56,
    0xFF04808868555C56,0xFF01878868554556,0x10704F8868556F56,0xFF1B4988685B6E56,
    0xFF9B6688685B6E56,0xFF9B808868556E56,0xFF1D738868557856,0xFF9B848868556E56,
    0xFF9B878868556E56,0x10776C8868551157,0xFF01738868554556,0xFF01798868554556,
    0xFF01808868554556,0xFF23738868554570,0xFF01848868554556,0xFF236688685B0D70,
    0x107E4F8868557856,0xFF1D4988685B7856,0xFF1D6688685B7856,0xFF1D798868557856,
    0xFF1D808868557856,0xFF1D848868557856,0xFF1D878868557856,0x10854F8868550D70,
    0xFF234988685B0D70,0xFF23738868550D70,0xFF23798868550D70,0xFF23808868550D70,
    0xFF23848868550D70,0xFF23878868550D70,0x108C4F8869551C10,0x108D4F8869554710,
    0xFF0449887D595C37,0xFF1049887D525C37,0xFF016688685B4537,0xFF0149887D654537,
    0xFF1E4988685B6E0A,0xFF1E878868556E0A,0x10944F8868554537,0xFF014988685B4537,
    0xFF01738868554537,0xFF01798868554537,0xFF01808868554537,0xFF01848868554537,
    0xFF01878868554537,0x109B4D8869556F0A,0xFF1E6688685B6E0A,0xFF0549887D59780A,
    0xFF1E49887D656E0A,0xFF1A49887D52780A,0xFF0573886855780A,0xFF0579886855780A,
    0x10A24F8869551137,0xFF0149887D594537,0xFF1049887D524537,0xFF014988685B0D37,
    0xFF016688685B0D37,0xFF0149887D650D37,0xFF01878868550D37,0x10A94F886955780A,
    0xFF054988685B780A,0xFF056688685B780A,0xFF0549887D65780A,0xFF0580886855780A,
    0xFF0584886855780A,0xFF0587886855780A,0x10B04E8869550D37,0xFF0149887D590D37,
    0xFF1049887D520D37,0xFF01738868550D37,0xFF01798868550D37,0xFF01808868550D37,
    0xFF01848868550D37,0x10B74F8868551C72,0x10B84F8868555D70,0xFF044988685B5C70,
    0xFF016688685B5C70,0xFF17738868556E70,0xFF17798868556E70,0xFF01848868555C70,
    0xFF01878868555C70,0x10BF4F8868554670,0xFF01738868555C70,0xFF234988685B4570,
    0xFF236688685B4570,0xFF01798868555C70,0xFF01808868555C70,0xFF23878868554570,
    0x10C64F8868556F70,0xFF154988685B6E70,0xFF176688685B6E70,0xFF17808868556E70,
    0xFFA8738868557870,0xFF17848868556E70,0xFF17878868556E70,0x10CD6C8868551172,
    0xFF23798868554570,0xFF23808868554570,0xFF0273886855457F,0xFF23848868554570,
    0xFF0279886855457F,0xFF026688685B0D7F,0x10D44F8868557870,0xFF154988685B7870,
    0xFFA86688685B7870,0xFFA8798868557870,0xFFA8808868557870,0xFFA8848868557870,
    0xFFA8878868557870,0x10DB4F8868550D7F,0xFF024988685B0D7F,0xFF02738868550D7F,
    0xFF02798868550D7F,0xFF02808868550D7F,0xFF02848868550D7F,0xFF02878868550D7F,
    0x10E24F886955170A,0x10E34C886955170A,0xFF1E6688685B5C0A,0xFF1E49887D596E0A,
    0xFF1A49887D526E0A,0xFF0C6688685B450A,0xFF8C6688685B0D0A,0xFF8C738868550D0A,
    0x10EA49886955460A,0xFF0C49887D595C0A,0xFF1A49887D525C0A,0xFF0C4988685B5C0A,
    0xFF0C4988685B450A,0xFF0C49887D655C0A,0xFF0C49887D65450A,0x10F149886955110A,
    0xFF0C49887D59450A,0xFF1A49887D52450A,0xFF0C49887D590D0A,0xFF1A49887D520D0A,
    0xFF0C4988685B0D0A,0xFF0C49887D650D0A,0x10F874886855170A,0xFF1E738868555C0A,
    0xFF1E798868555C0A,0xFF1E738868556E0A,0xFF0C73886855450A,0xFF0C79886855450A,
    0xFF8C798868550D0A,0x10FF7B886855170A,0xFF1E808868555C0A,0xFF1E798868556E0A,
    0xFF1E808868556E0A,0xFF0C80886855450A,0xFF8C808868550D0A,0xFF8C848868550D0A,
    0x110685886855170A,0xFF1E848868555C0A,0xFF1E878868555C0A,0xFF1E848868556E0A,
    0xFF0C84886855450A,0xFF0C87886855450A,0xFF8C878868550D0A,0x110D41886955487F,
    0x110E3B8869536F7F,0xFF023B8868526E7F,0xFF023B887D596E7F,0xFF023B887D526E7F,
    0xFF023B886852787F,0xFF023B887D59787F,0xFF023B887D52787F,0x1115748868555E7F,
    0xFF02738868555C7F,0xFF02798868555C7F,0xFF02738868556E7F,0xFF02798868556E7F,
    0xFF0273886855787F,0xFF0279886855787F,0x111C81886855487F,0xFF02808868555C7F,
    0xFF02848868555C7F,0xFF02808868556E7F,0xFF0280886855457F,0xFF0284886855457F,
    0xFF0280886855787F,0x112385886855487F,0xFF02878868555C7F,0xFF02848868556E7F,
    0xFF02878868556E7F,0xFF0287886855457F,0xFF0284886855787F,0xFF0287886855787F,
    0x112A3C88685B487F,0xFF024988685B5C7F,0xFF024988685B6E7F,0xFF024988685B457F,
    0xFF023B88685B6E7F,0xFF024988685B787F,0xFF023B88685B787F,0x11313D88695B487F,
    0xFF026688685B5C7F,0xFF026688685B6E7F,0xFF026688685B457F,0xFF026688685B787F,
    0xFF023B887D656E7F,0xFF023B887D65787F,0x1138208869551C1C,0x1139298869551C38,
    0x113A298869555C56,0xFF103B8868525C56,0xFF043B887D595C56,0xFF103B887D525C56,
    0xFF043B88685B5C56,0xFF042E887D535C56,0xFF0327887D535C56,0x1141298869554556,
    0xFF013B887D594556,0xFF013B88685B4556,0xFF012E887D534556,0xFF033B887D654556,
    0xFF032E887D654556,0xFF0327887D654556,0x1148298869555E38,0xFF103B8868526E56,
    0xFF033B887D655C56,0xFF032E887D655C56,0xFF0327887D655C56,0xFF103B887D527837,
    0xFF2227887D537837,0x114F298869551156,0xFF103B8868524556,0xFF103B887D524556,
    0xFF0327887D534556,0xFF033B887D650D56,0xFF032E887D650D56,0xFF0327887D650D56,
    0x1156298869557837,0xFFA03B887D597837,0xFFA03B88685B7837,0xFFA02E887D537837,
    0xFF223B887D657837,0xFF222E887D657837,0xFF2227887D657837,0x115D298869550D56,
    0xFF103B8868520D56,0xFF843B887D590D56,0xFF103B887D520D56,0xFF843B88685B0D56,
    0xFF842E887D530D56,0xFF0327887D530D56,0x1164298869551C57,0x1165298869555D57,
    0xFF1B3B887D596E56,0xFF103B8868525C70,0xFF103B887D526E56,0xFF1B3B88685B6E56,
    0xFF1B2E887D536E56,0xFF2227887D536E56,0x116C298869554570,0xFF233B887D594570,
    0xFF233B88685B4570,0xFF232E887D534570,0xFF233B887D654570,0xFF232E887D654570,
    0xFF2327887D654570,0x1173298869556F56,0xFF103B8868527856,0xFF223B887D656E56,
    0xFF103B887D527856,0xFF222E887D656E56,0xFF2227887D656E56,0xFF2227887D537856,
    0x117A298869551170,0xFF103B8868524570,0xFF103B887D524570,0xFF2327887D534570,
    0xFF233B887D650D70,0xFF232E887D650D70,0xFF2327887D650D70,0x1181298869557856,
    0xFF1D3B887D597856,0xFF1D3B88685B7856,0xFF1D2E887D537856,0xFF223B887D657856,
    0xFF222E887D657856,0xFF2227887D657856,0x1188298869550D70,0xFF103B8868520D70,
    0xFF233B887D590D70,0xFF103B887D520D70,0xFF233B88685B0D70,0xFF232E887D530D70,
    0xFF2327887D530D70,0x118F298869551C37,0x11902F887D531737,0xFF042E887D535C37,
    0xFF013B887D594537,0xFF052E887D536E37,0xFF012E887D534537,0xFF013B887D590D37,
    0xFF012E887D530D37,0x119729887D531737,0xFF103B887D525C37,0xFF103B887D526E37,
    0xFF0327887D535C37,0xFF2227887D536E37,0xFF8327887D534537,0xFF8327887D530D37,
    0x119E3B8869521C37,0xFF103B8868525C37,0xFF103B8868526E37,0xFF103B8868524537,
    0xFF103B887D524537,0xFF103B8868527837,0xFF103B887D520D37,0x11A53B88695B1737,
    0xFF043B887D595C37,0xFF043B88685B5C37,0xFF053B887D596E37,0xFF053B88685B6E37,
    0xFF013B88685B4537,0xFF013B88685B0D37,0x11AC2F887D651737,0xFF033B887D655C37,
    0xFF032E887D655C37,0xFF223B887D656E37,0xFF833B887D654537,0xFF222E887D656E37,
    0xFF833B887D650D37,0x11B328887D651737,0xFF0327887D655C37,0xFF832E887D654537,
    0xFF2227887D656E37,0xFF8327887D654537,0xFF832E887D650D37,0xFF8327887D650D37,
    0x11BA298869551C10,0x11BB29887D555D0A,0xFF0C3B887D595C0A,0xFF1A3B887D526E0A,
    0xFF033B887D655C0A,0xFF2227887D536E0A,0xFF032E887D655C0A,0xFF0327887D655C0A,
    0x11C229887D55460A,0xFF1A3B887D525C0A,0xFF0C2E887D535C0A,0xFF0327887D535C0A,
    0xFF833B887D65450A,0xFF832E887D65450A,0xFF8327887D65450A,0x11C929887D55110A,
    0xFF0C3B887D59450A,0xFF1A3B887D52450A,0xFF0C2E887D53450A,0xFF8327887D53450A,
    0xFF833B887D650D0A,0xFF832E887D650D0A,0x11D029887D556F0A,0xFF1E3B887D596E0A,
    0xFF1E2E887D536E0A,0xFF223B887D656E0A,0xFF1A3B887D52780A,0xFF222E887D656E0A,
    0xFF2227887D656E0A,0x11D729887D55780A,0xFF053B887D59780A,0xFF052E887D53780A,
    0xFF2227887D53780A,0xFF223B887D65780A,0xFF222E887D65780A,0xFF2227887D65780A,
    0x11DE298869550D10,0xFF103B8868520D37,0xFF0C3B887D590D0A,0xFF1A3B887D520D0A,
    0xFF0C2E887D530D0A,0xFF8327887D530D0A,0xFF8327887D650D0A,0x11E5208869551C72,
    0x11E6298869555D70,0xFF043B887D595C70,0xFF103B8868526E70,0xFF043B88685B5C70,
    0xFF043B887D655C70,0xFF042E887D655C70,0xFF0427887D655C70,0x11ED298869556E70,
    0xFF153B887D596E70,0xFF103B887D526E70,0xFF153B88685B6E70,0xFF152E887D536E70,
    0xFF1527887D536E70,0xFF1527887D656E70,0x11F4208869551472,0xFF103B887D525C70,
    0xFF042E887D535C70,0xFF0427887D535C70,0xFF023B88685B0D7F,0xFF0227887D650D7F,
    0xFF021D887D650D7F,0x11FB298869556F70,0xFF103B8868527870,0xFF153B887D656E70,
    0xFF103B887D527870,0xFF152E887D656E70,0xFF152E887D537870,0xFF1527887D537870,
    0x12022088695B7870,0xFF153B887D597870,0xFF153B88685B7870,0xFF153B887D657870,
    0xFF152E887D657870,0xFF1527887D657870,0xFF151D887D657870,0x1209208869530D7F,
    0xFF023B8868520D7F,0xFF023B887D590D7F,0xFF023B887D520D7F,0xFF022E887D530D7F,
    0xFF0227887D530D7F,0xFF021D887D530D7F,0x1210208869551C7F,0x12112088695B5C7F,
    0xFF023B887D595C7F,0xFF023B88685B5C7F,0xFF023B887D655C7F,0xFF022E887D655C7F,
    0xFF0227887D655C7F,0xFF021D887D655C7F,0x121820886955467F,0xFF023B8868525C7F,
    0xFF023B887D525C7F,0xFF022E887D535C7F,0xFF0227887D535C7F,0xFF021D887D535C7F,
    0xFF023B887D65457F,0x121F1F887D556E7F,0xFF022E887D536E7F,0xFF0227887D536E7F,
    0xFF021D887D536E7F,0xFF022E887D656E7F,0xFF0227887D656E7F,0xFF021D887D656E7F,
    0x122620886955457F,0xFF023B887D59457F,0xFF023B88685B457F,0xFF022E887D53457F,
    0xFF022E887D65457F,0xFF0227887D65457F,0xFF021D887D65457F,0x122D20886955117F,
    0xFF023B886852457F,0xFF023B887D52457F,0xFF0227887D53457F,0xFF021D887D53457F,
    0xFF023B887D650D7F,0xFF022E887D650D7F,0x12341F887D55787F,0xFF022E887D53787F,
    0xFF0227887D53787F,0xFF021D887D53787F,0xFF022E887D65787F,0xFF0227887D65787F,
    0xFF021D887D65787F,0x123B1C6226551C1C,0x123C3361265B1C1C,0x123D3E61265B1C56,
    0x123E3B61265B5D56,0xFF043B6142595C56,0xFF043B6142645C56,0xFF1B3B6142596E56,
    0xFF043B61255A5C56,0xFFB13B6142785C56,0xFFB13B6125785C56,0x12457361265B5D56,
    0xFF04736142595C56,0xFF04736142645C56,0xFF9B736142596E56,0xFF047361255A5C56,
    0xFF04736142785C56,0xFF04736125785C56,0x124C3B61265B1156,0xFF013B6142594556,
    0xFF013B6142644556,0xFF013B61255A4556,0xFFB13B6142784556,0xFFB13B6125784556,
    0xFFB13B6142780D56,0x12533B61265B6F56,0xFF1B3B6142646E56,0xFF1B3B61255A6E56,
    0xFF1D3B6142597856,0xFFB13B6142786E56,0xFF1D3B61255A7856,0xFFB13B6125786E56,
    0x125A7361265B1156,0xFF01736142594556,0xFF01736142644556,0xFF017361255A4556,
    0xFFB9736142784556,0xFFB9736125784556,0xFFB9736142780D56,0x12617361265B6F56,
    0xFF9B736142646E56,0xFF9B7361255A6E56,0xFF1D736142597856,0xFF1B736142786E56,
    0xFF1D7361255A7856,0xFF1B736125786E56,0x12683E61265B1C38,0x12693B61265B1C38,
    0xFF843B6142590D56,0xFF843B6142640D56,0xFF843B61255A0D56,0xFFA03B6142787837,
    0xFFB13B6125780D56,0xFFA03B6125787837,0x12707361265B1C38,0xFF81736142590D56,
    0xFF81736142640D56,0xFF817361255A0D56,0xFF20736142787837,0xFFB9736125780D56,
    0xFF20736125787837,0x12773B61265B5D37,0xFF043B6142645C37,0xFF053B6142596E37,
    0xFF043B61255A5C37,0xFF053B61255A6E37,0xFF043B6142785C37,0xFF043B6125785C37,
    0x127E7361265B5D37,0xFF04736142645C37,0xFF05736142596E37,0xFF047361255A5C37,
    0xFF057361255A6E37,0xFFB1736142785C37,0xFFB1736125785C37,0x12853B61265B6F37,
    0xFF053B6142646E37,0xFFA03B6142597837,0xFFA03B6142647837,0xFF053B6142786E37,
    0xFFA03B61255A7837,0xFF053B6125786E37,0x128C7361265B6F37,0xFF05736142646E37,
    0xFF20736142597837,0xFF20736142647837,0xFFB1736142786E37,0xFF207361255A7837,
    0xFFB1736125786E37,0x12933E61265B1C57,0x12943E6126657856,0xFF1D3B6142647856,
    0xFF1D736142647856,0xFF1D3B6142787856,0xFF1D736142787856,0xFF1D3B6125787856,
    0xFF1D736125787856,0x129B3B61265B5D70,0xFF043B6142645C70,0xFF153B6142596E70,
    0xFF043B61255A5C70,0xFF153B61255A6E70,0xFF043B6142785C70,0xFF043B6125785C70,
    0x12A23E61265B4670,0xFF043B6142595C70,0xFF233B6142644570,0xFF233B61255A4570,
    0xFF237361255A4570,0xFF243B6142784570,0xFF243B6125784570,0x12A93E61265B1170,
    0xFF233B6142594570,0xFF23736142594570,0xFF243B6142780D70,0xFF24736142780D70,
    0xFF243B6125780D70,0xFF24736125780D70,0x12B03B61265B6F70,0xFF153B6142646E70,
    0xFF153B6142597870,0xFF153B6142647870,0xFF043B6142786E70,0xFF153B61255A7870,
    0xFF043B6125786E70,0x12B73E61265A0D70,0xFF233B6142590D70,0xFF23736142590D70,
    0xFF233B6142640D70,0xFF23736142640D70,0xFF233B61255A0D70,0xFF237361255A0D70,
    0x12BE3E61265B1C10,0x12BF3E61265B4710,0xFF043B6142595C37,0xFF04736142595C37,
    0xFFB13B6142784537,0xFFB1736142784537,0xFFB23B6125786E0A,0xFFB2736125786E0A,
    0x12C63E61265B4537,0xFF013B6142644537,0xFF01736142644537,0xFF013B61255A4537,
    0xFF017361255A4537,0xFFB13B6125784537,0xFFB1736125784537,0x12CD3E61265B1137,
    0xFF013B6142594537,0xFF01736142594537,0xFFB13B6142780D37,0xFFB9736142780D37,
    0xFFB13B6125780D37,0xFFB9736125780D37,0x12D43E61265B6F0A,0xFF053B614259780A,
    0xFF0573614259780A,0xFF1E3B6142786E0A,0xFF1E736142786E0A,0xFFB23B61255A780A,
    0xFFB27361255A780A,0x12DB3E612665780A,0xFF053B614264780A,0xFF0573614264780A,
    0xFF1E3B614278780A,0xFF1E73614278780A,0xFFB23B612578780A,0xFFB273612578780A,
    0x12E23E61265A0D37,0xFF013B6142590D37,0xFF01736142590D37,0xFF013B6142640D37,
    0xFF01736142640D37,0xFF013B61255A0D37,0xFF017361255A0D37,0x12E92F61265B1C72,
    0x12EA2F61265B5C7F,0xFF253B6142645C7F,0xFF253B61255A5C7F,0xFF253B6142785C7F,
    0xFF252E6142655C7F,0xFF253B6125785C7F,0xFF252E6125655C7F,0x12F12F61265B467F,
    0xFF023B6142595C7F,0xFF253B614264457F,0xFF252E6125595C7F,0xFF253B61255A457F,
    0xFF253B614278457F,0xFF253B612578457F,0x12F82F61265B6E7F,0xFF023B6142596E7F,
    0xFF263B6142646E7F,0xFF263B61255A6E7F,0xFF262E6125596E7F,0xFF262E6142656E7F,
    0xFF262E6125656E7F,0x12FF3B61265B117F,0xFF023B614259457F,0xFF023B6142590D7F,
    0xFF253B6142640D7F,0xFF253B61255A0D7F,0xFF253B6142780D7F,0xFF253B6125780D7F,
    0x13062F61265B6F7F,0xFF023B614259787F,0xFF263B614264787F,0xFF263B6142786E7F,
    0xFF263B61255A787F,0xFF262E612559787F,0xFF263B6125786E7F,0x130D2F6126657872,
    0xFF153B6142787870,0xFF153B6125787870,0xFF263B614278787F,0xFF262E614265787F,
    0xFF263B612578787F,0xFF262E612565787F,0x13143E61265B170A,0x13153B61265B5D0A,
    0xFF0C3B6142645C0A,0xFF1E3B6142596E0A,0xFF1E3B6142646E0A,0xFFB23B61255A6E0A,
    0xFF0C3B6142785C0A,0xFFB23B6125785C0A,0x131C7361265B5D0A,0xFF1E736142645C0A,
    0xFF1E736142596E0A,0xFF1E736142646E0A,0xFFB27361255A6E0A,0xFF0C736142785C0A,
    0xFFB2736125785C0A,0x13233B61265B460A,0xFF0C3B6142595C0A,0xFFB23B61255A5C0A,
    0xFF0C3B614264450A,0xFFB33B61255A450A,0xFF0C3B614278450A,0xFFB33B612578450A,
    0x132A7361265B460A,0xFF1E736142595C0A,0xFFB27361255A5C0A,0xFF0C73614264450A,
    0xFFB37361255A450A,0xFF0C73614278450A,0xFFB373612578450A,0x13313B61265B110A,
    0xFF0C3B614259450A,0xFF0C3B6142590D0A,0xFF0C3B6142640D0A,0xFFB33B61255A0D0A,
    0xFF0C3B6142780D0A,0xFFB33B6125780D0A,0x13387361265B110A,0xFF0C73614259450A,
    0xFF8C736142590D0A,0xFF8C736142640D0A,0xFFB37361255A0D0A,0xFF8C736142780D0A,
    0xFFB3736125780D0A,0x133F766126551C1C,0x13407A6126551C38,0x13417A6126555C56,
    0xFF04796125595C56,0xFF04796142655C56,0xFF04806125535C56,0xFF04806142655C56,
    0xFFB2796125655C56,0xFFB5806125655C56,0x13487A6125556E38,0xFF9B796125596E56,
    0xFF9B806125536E56,0xFF05796125596E37,0xFFB2796125656E56,0xFFB2796125656E37,
    0xFFB5806125656E37,0x134F7A6126554556,0xFF01796125594556,0xFFB9796142654556,
    0xFF01806125534556,0xFFB9806142654556,0xFFB3796125654556,0xFFB5806125654556,
    0x13567A6126556F38,0xFF1B796142656E56,0xFF1B806142656E56,0xFFB1796142656E37,
    0xFFB1806142656E37,0xFFB5806125656E56,0xFF20806125537837,0x135D7A6126557838,
    0xFF1D806125537856,0xFF20796125597837,0xFF20796142657837,0xFF20806142657837,
    0xFFB2796125657837,0xFFB5806125657837,0x13647A6126550D56,0xFF81796125590D56,
    0xFFB9796142650D56,0xFF81806125530D56,0xFFB9806142650D56,0xFFB3796125650D56,
    0xFFB5806125650D56,0x136B756126551C57,0x136C7561265B5C70,0xFF01736142645C70,
    0xFF017361255A5C70,0xFF04796142655C70,0xFF04806142655C70,0xFF04796125655C70,
    0xFFB6806125655C70,0x1373756126555D70,0xFF17736142596E70,0xFF177361255A6E70,
    0xFF04736142785C70,0xFF17796125596E70,0xFF17806125536E70,0xFF04736125785C70,
    0x137A756126554670,0xFF01736142595C70,0xFF01796125595C70,0xFF01806125535C70,
    0xFF24736142784570,0xFF24806142654570,0xFF24736125784570,0x1381756126554570,
    0xFF23736142644570,0xFF23796125594570,0xFF24796142654570,0xFF23806125534570,
    0xFF24796125654570,0xFFB6806125654570,0x13887561265B6F57,0xFF17736142646E70,
    0xFF1D796125597856,0xFF1D796142657856,0xFF1D806142657856,0xFFB2796125657856,
    0xFFB5806125657856,0x138F7A6126550D70,0xFF23796125590D70,0xFF24796142650D70,
    0xFF23806125530D70,0xFF24806142650D70,0xFF24796125650D70,0xFFB6806125650D70,
    0x13967B6126551C10,0x13977B6125555C37,0xFF04796125595C37,0xFF04806125535C37,
    0xFF04846125595C37,0xFFB2796125655C37,0xFFB5806125655C37,0xFFB2846125655C37,
    0x139E7B6126654537,0xFFB1796142654537,0xFFB1806142654537,0xFFB3796125654537,
    0xFFB1846142654537,0xFFB5806125654537,0xFFB3846125654537,0x13A57B6126555E10,
    0xFFB1796142655C37,0xFFB1806142655C37,0xFF05806125536E37,0xFFB279612559780A,
    0xFFB280612553780A,0xFFB284612559780A,0x13AC7B6126551137,0xFF01796125594537,
    0xFF01806125534537,0xFF01846125594537,0xFFB9796142650D37,0xFFB9806142650D37,
    0xFFB9846142650D37,0x13B37B612665780A,0xFF1E79614265780A,0xFF1E80614265780A,
    0xFFB279612565780A,0xFF1E84614265780A,0xFFB580612565780A,0xFFB284612565780A,
    0x13BA7B6125550D37,0xFF01796125590D37,0xFF01806125530D37,0xFF01846125590D37,
    0xFFB3796125650D37,0xFFB5806125650D37,0xFFB3846125650D37,0x13C1756126551C72,
    0x13C2756126654772,0xFF04796142656E70,0xFF04806142656E70,0xFF04736125786E70,
    0xFF04796125656E70,0xFFA579614265457F,0xFFB6806125656E70,0x13C97561265B457F,
    0xFF0273614259457F,0xFFA573614264457F,0xFFA57361255A457F,0xFFA579612559457F,
    0xFFA579612565457F,0xFFA580612565457F,0x13D0756126556F70,0xFFA8736142597870,
    0xFFA8736142647870,0xFF04736142786E70,0xFFA87361255A7870,0xFFA8796125597870,
    0xFFA8806125537870,0x13D775612655117F,0xFFA580612553457F,0xFFA5736142780D7F,
    0xFFA5796142650D7F,0xFFA5806142650D7F,0xFFA5736125780D7F,0xFFA5806125650D7F,
    0x13DE756126657870,0xFF15736142787870,0xFF15796142657870,0xFF15806142657870,
    0xFF15736125787870,0xFF15796125657870,0xFFB6806125657870,0x13E5756126550D7F,
    0xFF02736142590D7F,0xFFA5736142640D7F,0xFFA57361255A0D7F,0xFFA5796125590D7F,
    0xFFA5806125530D7F,0xFFA5796125650D7F,0x13EC7B612655170A,0x13ED7961265B5D0A,
    0xFFB2796125595C0A,0xFF0C796142655C0A,0xFFB2796125596E0A,0xFF1E796142656E0A,
    0xFFB2796125655C0A,0xFFB2796125656E0A,0x13F47961265B110A,0xFFB379612559450A,
    0xFF0C79614265450A,0xFFB3796125590D0A,0xFFB379612565450A,0xFF8C796142650D0A,
    0xFFB3796125650D0A,0x13FB806126555D0A,0xFFB2806125535C0A,0xFF0C806142655C0A,
    0xFFB2806125536E0A,0xFF1E806142656E0A,0xFFB5806125655C0A,0xFFB5806125656E0A,
    0x140280612655110A,0xFFB380612553450A,0xFF0C80614265450A,0xFFB580612565450A,
    0xFFB3806125530D0A,0xFF8C806142650D0A,0xFFB5806125650D0A,0x14098461265B5D0A,
    0xFFB2846125595C0A,0xFF0C846142655C0A,0xFFB2846125596E0A,0xFF1E846142656E0A,
    0xFFB2846125655C0A,0xFFB2846125656E0A,0x14108461265B110A,0xFFB384612559450A,
    0xFF0C84614265450A,0xFFB3846125590D0A,0xFFB384612565450A,0xFF8C846142650D0A,
    0xFFB3846125650D0A,0x141775612655487F,0x14187561265B5C7F,0xFF25736142645C7F,
    0xFF257361255A5C7F,0xFF25796142655C7F,0xFF25806142655C7F,0xFF25796125655C7F,
    0xFF25806125655C7F,0x141F756126555D7F,0xFF02736142596E7F,0xFF267361255A6E7F,
    0xFF25736142785C7F,0xFF26796125596E7F,0xFF26806125536E7F,0xFF25736125785C7F,
    0x142675612655467F,0xFF02736142595C7F,0xFF25796125595C7F,0xFF25806125535C7F,
    0xFFA573614278457F,0xFFA580614265457F,0xFFA573612578457F,0x142D756126656E7F,
    0xFF26736142646E7F,0xFF26796142656E7F,0xFF26806142656E7F,0xFF26736125786E7F,
    0xFF26796125656E7F,0xFF26806125656E7F,0x1434756126556F7F,0xFF0273614259787F,
    0xFF2673614264787F,0xFF26736142786E7F,0xFF267361255A787F,0xFF2679612559787F,
    0xFF2680612553787F,0x143B75612665787F,0xFF2673614278787F,0xFF2679614265787F,
    0xFF2680614265787F,0xFF2673612578787F,0xFF2679612565787F,0xFF2680612565787F,
    0x14420B6126551C1C,0x14430B6126551C38,0x14440B6126655C56,0xFFB12E6142655C56,
    0xFF04066142645C56,0xFFB1276142655C56,0xFFB22E6125655C56,0xFFB11D6142655C56,
    0xFFB4276125655C56,0x144B0B61265B4656,0xFF042E6125595C56,0xFF04066142595C56,
    0xFFB12E6142654556,0xFFB1276142654556,0xFFB11D6142654556,0xFFB1066142784556,
    0x14520B6126555E38,0xFF1B2E6125596E56,0xFF1B066142596E56,0xFF1B276125536E56,
    0xFFB1066142785C56,0xFFA02E6125597837,0xFFA0066142597837,0x14590B61265B1156,
    0xFF012E6125594556,0xFF01066142594556,0xFF01066142644556,0xFFB32E6125654556,
    0xFFB12E6142650D56,0xFFB1066142780D56,0x14600B6126657837,0xFFA02E6142657837,
    0xFFA0066142647837,0xFFA0276142657837,0xFFB22E6125657837,0xFFA01D6142657837,
    0xFFA0066142787837,0x14670B61265B0D56,0xFF842E6125590D56,0xFF84066142590D56,
    0xFF84066142640D56,0xFFB1276142650D56,0xFFB32E6125650D56,0xFFB11D6142650D56,
    0x146E0B6126551C57,0x146F0B6126654757,0xFF1B066142646E56,0xFFB1276142656E56,
    0xFFB22E6125656E56,0xFFB11D6142656E56,0xFF242E6142654570,0xFFB4276125656E56,
    0x14760B61265B4570,0xFF23066142594570,0xFF23066142644570,0xFF24276142654570,
    0xFF242E6125654570,0xFF241D6142654570,0xFFB6276125654570,0x147D0B6126556F56,
    0xFFB12E6142656E56,0xFF1D2E6125597856,0xFF1D066142597856,0xFF1D276125537856,
    0xFF1D066142647856,0xFFB1066142786E56,0x14840B6126551170,0xFF232E6125594570,
    0xFF23276125534570,0xFF242E6142650D70,0xFF24276142650D70,0xFF241D6142650D70,
    0xFF24066142780D70,0x148B0B6126657856,0xFF1D2E6142657856,0xFF1D276142657856,
    0xFFB22E6125657856,0xFF1D1D6142657856,0xFFB4276125657856,0xFF1D066142787856,
    0x14920B6126550D70,0xFF232E6125590D70,0xFF23066142590D70,0xFF23276125530D70,
    0xFF23066142640D70,0xFF242E6125650D70,0xFFB6276125650D70,0x14990B61265B1C10,
    0x149A0B6126655C37,0xFF042E6142655C37,0xFF04066142645C37,0xFF04276142655C37,
    0xFFB22E6125655C37,0xFF041D6142655C37,0xFF04066142785C37,0x14A10B61265B4637,
    0xFF042E6125595C37,0xFF04066142595C37,0xFFB12E6142654537,0xFFB1276142654537,
    0xFFB11D6142654537,0xFFB1066142784537,0x14A80B61265B6E37,0xFF052E6125596E37,
    0xFF05066142596E37,0xFF05066142646E37,0xFF05276142656E37,0xFFB22E6125656E37,
    0xFF051D6142656E37,0x14AF0B61265B1137,0xFF012E6125594537,0xFF01066142594537,
    0xFF01066142644537,0xFFB32E6125654537,0xFFB12E6142650D37,0xFFB1066142780D37,
    0x14B60B6142656F10,0xFF052E6142656E37,0xFF05066142786E37,0xFF1E2E614265780A,
    0xFF1E27614265780A,0xFF1E1D614265780A,0xFF1E06614278780A,0x14BD0B61265B0D37,
    0xFF012E6125590D37,0xFF01066142590D37,0xFF01066142640D37,0xFFB1276142650D37,
    0xFFB32E6125650D37,0xFFB11D6142650D37,0x14C40B6126551C72,0x14C50B6126555C70,
    0xFF042E6125595C70,0xFF04066142595C70,0xFF04276125535C70,0xFF04066142645C70,
    0xFF042E6125655C70,0xFFB6276125655C70,0x14CC0B6126555D70,0xFF042E6142655C70,
    0xFF152E6125596E70,0xFF04276142655C70,0xFF15276125536E70,0xFF041D6142655C70,
    0xFF04066142785C70,0x14D30B61265B6E70,0xFF15066142596E70,0xFF15066142646E70,
    0xFF04276142656E70,0xFF042E6125656E70,0xFF041D6142656E70,0xFFB6276125656E70,
    0x14DA0B6126556F70,0xFF042E6142656E70,0xFF152E6125597870,0xFF15066142597870,
    0xFF15276125537870,0xFF15066142647870,0xFF04066142786E70,0x14E10B6126551172,
    0xFF24066142784570,0xFF252E6125590D7F,0xFF02066142590D7F,0xFF25276125530D7F,
    0xFF25066142640D7F,0xFF25276125650D7F,0x14E80B6126657870,0xFF152E6142657870,
    0xFF15276142657870,0xFF152E6125657870,0xFF151D6142657870,0xFFB6276125657870,
    0xFF15066142787870,0x14EF0B61265B1C0A,0x14F02E61265B140A,0xFFB22E6125595C0A,
    0xFFB32E612559450A,0xFF0C2E614265450A,0xFFB22E6125655C0A,0xFFB32E612565450A,
    0xFF0C2E6142650D0A,0x14F72E61265B5E0A,0xFF0C2E6142655C0A,0xFFB22E6125596E0A,
    0xFF1E2E6142656E0A,0xFFB22E612559780A,0xFFB22E6125656E0A,0xFFB22E612565780A,
    0x14FE2861265B170A,0xFF0C276142655C0A,0xFF1E276142656E0A,0xFF0C27614265450A,
    0xFFB32E6125590D0A,0xFF0C276142650D0A,0xFFB32E6125650D0A,0x15050861425B1C0A,
    0xFF0C1D6142655C0A,0xFF1E1D6142656E0A,0xFF0C1D614265450A,0xFF0506614259780A,
    0xFF0506614264780A,0xFF0C1D6142650D0A,0x150C0661425B5D0A,0xFF0C066142595C0A,
    0xFF0C066142645C0A,0xFF1E066142596E0A,0xFF1E066142646E0A,0xFF0C066142785C0A,
    0xFF1E066142786E0A,0x15130661425B110A,0xFF0C06614259450A,0xFF0C06614264450A,
    0xFF0C066142590D0A,0xFF0C066142640D0A,0xFF0C06614278450A,0xFF0C066142780D0A,
    0x151A0B6126551C7F,0x151B096126555D7F,0xFF25276142655C7F,0xFF02066142596E7F,
    0xFF26276125536E7F,0xFF251D6142655C7F,0xFF25276125655C7F,0xFF25066142785C7F,
    0x15220B612655467F,0xFF02066142595C7F,0xFF25276125535C7F,0xFF25066142645C7F,
    0xFF252E614265457F,0xFF2527614265457F,0xFF2506614278457F,0x15290B61265B457F,
    0xFF252E612559457F,0xFF0206614259457F,0xFF2506614264457F,0xFF252E612565457F,
    0xFF251D614265457F,0xFF2527612565457F,0x15300B612655117F,0xFF2527612553457F,
    0xFF252E6142650D7F,0xFF25276142650D7F,0xFF252E6125650D7F,0xFF251D6142650D7F,
    0xFF25066142780D7F,0x1537096126556F7F,0xFF26066142646E7F,0xFF26276142656E7F,
    0xFF261D6142656E7F,0xFF26276125656E7F,0xFF2627612553787F,0xFF26066142786E7F,
    0x153E0961265B787F,0xFF0206614259787F,0xFF2606614264787F,0xFF2627614265787F,
    0xFF261D614265787F,0xFF2627612565787F,0xFF2606614278787F,0x15451C62265B1C1C,
    0x15461B62265B1C1C,0x15471B62265B5D39,0xFFB1846142655C37,0xFF05846125596E37,
    0xFFB1846142656E37,0xFFB2846125656E37,0xFF150688255A6E70,0xFF04068825786E70,
    0x154E1D88255B4737,0xFF041D8825595C37,0xFF051D8825596E37,0xFF011D8825594537,
    0xFFB21D8825655C37,0xFFB21D8825656E37,0xFFB31D8825654537,0x15551D88255B1C10,
    0xFFB21D8825596E0A,0xFF011D8825590D37,0xFFB21D882559780A,0xFFB21D8825656E0A,
    0xFFB31D8825650D37,0xFFB21D882565780A,0x155C1D88255B140A,0xFFB21D8825595C0A,
    0xFFB31D882559450A,0xFFB21D8825655C0A,0xFFB31D8825590D0A,0xFFB31D882565450A,
    0xFFB31D8825650D0A,0x15630688255B1C72,0xFF250688255A457F,0xFF150688255A7870,
    0xFF250688255A0D7F,0xFF2506882578457F,0xFF15068825787870,0xFF25068825780D7F,
    0x156A0688255B5E7F,0xFF250688255A5C7F,0xFF260688255A6E7F,0xFF25068825785C7F,
    0xFF260688255A787F,0xFF26068825786E7F,0xFF2606882578787F,0x15718461265B1C3A,
    0x15728461265B5D57,0xFF04846142655C56,0xFF9B846125596E56,0xFF04846142655C70,
    0xFF17846125596E70,0xFFB2846125656E56,0xFF04846125655C70,0x15798461265B4657,
    0xFF04846125595C56,0xFF01846125595C70,0xFFB9846142654556,0xFFB2846125655C56,
    0xFF24846142654570,0xFF24846125654570,0x15808461265B6F39,0xFF1B846142656E56,
    0xFF1D846125597856,0xFF04846142656E70,0xFF20846125597837,0xFFA8846125597870,
    0xFF04846125656E70,0x15878461265B1158,0xFF01846125594556,0xFF23846125594570,
    0xFFB3846125654556,0xFFB9846142650D56,0xFF24846142650D70,0xFFA5846142650D7F,
    0x158E846126657839,0xFF1D846142657856,0xFF20846142657837,0xFFB2846125657856,
    0xFF15846142657870,0xFFB2846125657837,0xFF15846125657870,0x15958461255B0D58,
    0xFF81846125590D56,0xFF23846125590D70,0xFFB3846125650D56,0xFFA5846125590D7F,
    0xFF24846125650D70,0xFFA5846125650D7F,0x159C8561265B1C1C,0x159D8761265B5D0A,
    0xFF1E876142645C0A,0xFF1E876142596E0A,0xFF1E876142646E0A,0xFFB28761255A6E0A,
    0xFF0C876142785C0A,0xFFB2876125785C0A,0x15A48761265B460A,0xFF1E876142595C0A,
    0xFFB28761255A5C0A,0xFF0C87614264450A,0xFFB38761255A450A,0xFF0C87614278450A,
    0xFFB387612578450A,0x15AB8761265B110A,0xFF0C87614259450A,0xFF8C876142590D0A,
    0xFF8C876142640D0A,0xFFB38761255A0D0A,0xFF8C876142780D0A,0xFFB3876125780D0A,
    0x15B28761265B6F0A,0xFF0587614259780A,0xFF0587614264780A,0xFF1E876142786E0A,
    0xFFB28761255A780A,0xFFB2876125786E0A,0xFFB287612578780A,0x15B98461265B467F,
    0xFF25846125595C7F,0xFF25846142655C7F,0xFFA584612559457F,0xFFA584614265457F,
    0xFF25846125655C7F,0xFFA584612565457F,0x15C08461265B6F7F,0xFF26846125596E7F,
    0xFF26846142656E7F,0xFF2684612559787F,0xFF26846125656E7F,0xFF2684614265787F,
    0xFF2684612565787F,0x15C70688255B1C19,0x15C80688255B5E56,0xFF040688255A5C56,
    0xFF1B0688255A6E56,0xFFB1068825785C56,0xFF1D0688255A7856,0xFFB1068825786E56,
    0xFF1D068825787856,0x15CF0688255B1C38,0xFF010688255A4556,0xFF840688255A0D56,
    0xFFB1068825784556,0xFFA00688255A7837,0xFFB1068825780D56,0xFFA0068825787837,
    0x15D60688255B4737,0xFF040688255A5C37,0xFF050688255A6E37,0xFF010688255A4537,
    0xFF04068825785C37,0xFF05068825786E37,0xFFB1068825784537,0x15DD0688255B1470,
    0xFF040688255A5C70,0xFF230688255A4570,0xFF04068825785C70,0xFF230688255A0D70,
    0xFF24068825784570,0xFF24068825780D70,0x15E40688255B1C10,0xFFB20688255A6E0A,
    0xFF010688255A0D37,0xFFB20688255A780A,0xFFB2068825786E0A,0xFFB1068825780D37,
    0xFFB206882578780A,0x15EB0688255B140A,0xFFB20688255A5C0A,0xFFB30688255A450A,
    0xFFB2068825785C0A,0xFFB30688255A0D0A,0xFFB306882578450A,0xFFB3068825780D0A,
    0x15F28761265B1C15,0x15F38761265B5D37,0xFF05876142596E37,0xFF05876142646E37,
    0xFF058761255A6E37,0xFFB1876142785C37,0xFFB1876142786E37,0xFFB1876125786E37,
    0x15FA8761265B4638,0xFF04876142595C37,0xFF04876142645C37,0xFF048761255A5C37,
    0xFFB9876142784556,0xFFB1876142784537,0xFFB1876125785C37,0x16018761265B4538,
    0xFF01876142644556,0xFF018761255A4556,0xFF01876142644537,0xFF018761255A4537,
    0xFFB9876125784556,0xFFB1876125784537,0x16088761265B1138,0xFF01876142594556,
    0xFF01876142594537,0xFFB9876142780D56,0xFFB9876142780D37,0xFFB9876125780D56,
    0xFFB9876125780D37,0x160F8761265B7810,0xFF20876142597837,0xFF20876142647837,
    0xFF208761255A7837,0xFF20876142787837,0xFF20876125787837,0xFF1E87614278780A,
    0x16168761265A0D38,0xFF81876142590D56,0xFF81876142640D56,0xFF01876142590D37,
    0xFF818761255A0D56,0xFF01876142640D37,0xFF018761255A0D37,0x161D8761265B1C58,
    0x161E8761265B5D56,0xFF04876142595C56,0xFF04876142645C56,0xFF9B876142596E56,
    0xFF048761255A5C56,0xFF04876142785C56,0xFF04876125785C56,0x16258761265B6F56,
    0xFF9B876142646E56,0xFF9B8761255A6E56,0xFF1D876142597856,0xFF1B876142786E56,
    0xFF1D8761255A7856,0xFF1B876125786E56,0x162C8761265B1C57,0xFF1D876142647856,
    0xFF23876142590D70,0xFF1D876142787856,0xFF238761255A0D70,0xFF1D876125787856,
    0xFF24876125780D70,0x16338761255B4770,0xFF018761255A5C70,0xFF178761255A6E70,
    0xFF238761255A4570,0xFF04876125785C70,0xFF04876125786E70,0xFF24876125784570,
    0x163A8761255B1C72,0xFFA58761255A457F,0xFFA88761255A7870,0xFFA58761255A0D7F,
    0xFFA587612578457F,0xFF15876125787870,0xFFA5876125780D7F,0x16418761255B5E7F,
    0xFF258761255A5C7F,0xFF268761255A6E7F,0xFF25876125785C7F,0xFF268761255A787F,
    0xFF26876125786E7F,0xFF2687612578787F,0x1648228825551C1C,0x1649228825551C38,
    0x164A228825554656,0xFF043B88255A5C56,0xFF042E8825595C56,0xFF04278825535C56,
    0xFF041D8825595C56,0xFFB13B8825784556,0xFFB9738825784556,0x16512288255B4556,
    0xFF013B88255A4556,0xFF017388255A4556,0xFF012E8825594556,0xFFB32E8825654556,
    0xFFB4278825654556,0xFFB31D8825654556,0x1658228825555E38,0xFF047388255A5C56,
    0xFFA03B88255A7837,0xFFB21D8825655C56,0xFFA02E8825597837,0xFFA0278825537837,
    0xFFA01D8825597837,0x165F228825551156,0xFF01278825534556,0xFF011D8825594556,
    0xFFB13B8825780D56,0xFFB9738825780D56,0xFFB32E8825650D56,0xFFB4278825650D56,
    0x16662288255B7837,0xFF207388255A7837,0xFFA03B8825787837,0xFF20738825787837,
    0xFFB22E8825657837,0xFFB4278825657837,0xFFB21D8825657837,0x166D228825550D56,
    0xFF843B88255A0D56,0xFF817388255A0D56,0xFF842E8825590D56,0xFF84278825530D56,
    0xFF841D8825590D56,0xFFB31D8825650D56,0x1674228825551C57,0x1675228825555D56,
    0xFF1B278825536E56,0xFFB13B8825785C56,0xFF04738825785C56,0xFFB22E8825655C56,
    0xFF1B1D8825596E56,0xFFB4278825655C56,0x167C2288255B6E56,0xFF1B3B88255A6E56,
    0xFF9B7388255A6E56,0xFF1B2E8825596E56,0xFFB22E8825656E56,0xFFB4278825656E56,
    0xFFB21D8825656E56,0x1683228825556F56,0xFF1D3B88255A7856,0xFF1D2E8825597856,
    0xFFB13B8825786E56,0xFF1B738825786E56,0xFF1D278825537856,0xFF1D1D8825597856,
    0x168A2288255B7856,0xFF1D7388255A7856,0xFF1D3B8825787856,0xFF1D738825787856,
    0xFFB22E8825657856,0xFFB4278825657856,0xFFB21D8825657856,0x1691228825551170,
    0xFF23278825534570,0xFF231D8825594570,0xFF243B8825780D70,0xFF24738825780D70,
    0xFF242E8825650D70,0xFFB6278825650D70,0x1698228825550D70,0xFF233B88255A0D70,
    0xFF237388255A0D70,0xFF232E8825590D70,0xFF23278825530D70,0xFF231D8825590D70,
    0xFF241D8825650D70,0x169F2C8825551C10,0x16A02C88255B5C37,0xFF043B88255A5C37,
    0xFF047388255A5C37,0xFF043B8825785C37,0xFFB1738825785C37,0xFFB22E8825655C37,
    0xFFB4278825655C37,0x16A72C8825554637,0xFF042E8825595C37,0xFF04278825535C37,
    0xFFB13B8825784537,0xFFB1738825784537,0xFFB32E8825654537,0xFFB4278825654537,
    0x16AE2C8825556E37,0xFF053B88255A6E37,0xFF057388255A6E37,0xFF052E8825596E37,
    0xFF05278825536E37,0xFFB22E8825656E37,0xFFB4278825656E37,0x16B52C8825551137,
    0xFF013B88255A4537,0xFF017388255A4537,0xFF012E8825594537,0xFF01278825534537,
    0xFFB13B8825780D37,0xFFB9738825780D37,0x16BC2C8825656F10,0xFF053B8825786E37,
    0xFFB1738825786E37,0xFFB23B882578780A,0xFFB273882578780A,0xFFB22E882565780A,
    0xFFB427882565780A,0x16C32C8825550D37,0xFF013B88255A0D37,0xFF017388255A0D37,
    0xFF012E8825590D37,0xFF01278825530D37,0xFFB32E8825650D37,0xFFB4278825650D37,
    0x16CA228825551C72,0x16CB208825555C70,0xFF043B88255A5C70,0xFF042E8825595C70,
    0xFF04278825535C70,0xFF041D8825595C70,0xFFB6278825655C70,0xFF041D8825655C70,
    0x16D2208825555D70,0xFF153B88255A6E70,0xFF152E8825596E70,0xFF15278825536E70,
    0xFF043B8825785C70,0xFF042E8825655C70,0xFF151D8825596E70,0x16D92288255B4570,
    0xFF237388255A4570,0xFF243B8825784570,0xFF24738825784570,0xFF242E8825654570,
    0xFFB6278825654570,0xFF241D8825654570,0x16E0208825541172,0xFF233B88255A4570,
    0xFF232E8825594570,0xFF253B88255A0D7F,0xFF252E8825590D7F,0xFF25278825530D7F,
    0xFF251D8825590D7F,0x16E7208825556F70,0xFF043B8825786E70,0xFF042E8825656E70,
    0xFFB6278825656E70,0xFF15278825537870,0xFF151D8825597870,0xFF041D8825656E70,
    0x16EE2088255B7870,0xFF153B88255A7870,0xFF152E8825597870,0xFF153B8825787870,
    0xFF152E8825657870,0xFFB6278825657870,0xFF151D8825657870,0x16F52C8825551C0A,
    0x16F62C88255B5C0A,0xFFB23B88255A5C0A,0xFFB27388255A5C0A,0xFFB23B8825785C0A,
    0xFFB2738825785C0A,0xFFB22E8825655C0A,0xFFB4278825655C0A,0x16FD2C882555460A,
    0xFFB22E8825595C0A,0xFFB2278825535C0A,0xFFB33B882578450A,0xFFB373882578450A,
    0xFFB32E882565450A,0xFFB427882565450A,0x17042C8825556E0A,0xFFB23B88255A6E0A,
    0xFFB27388255A6E0A,0xFFB22E8825596E0A,0xFFB2278825536E0A,0xFFB22E8825656E0A,
    0xFFB4278825656E0A,0x170B2C882555110A,0xFFB33B88255A450A,0xFFB37388255A450A,
    0xFFB32E882559450A,0xFFB327882553450A,0xFFB33B8825780D0A,0xFFB3738825780D0A,
    0x17122C8825556F0A,0xFFB23B88255A780A,0xFFB27388255A780A,0xFFB22E882559780A,
    0xFFB23B8825786E0A,0xFFB2738825786E0A,0xFFB227882553780A,0x17192C8825550D0A,
    0xFFB33B88255A0D0A,0xFFB37388255A0D0A,0xFFB32E8825590D0A,0xFFB3278825530D0A,
    0xFFB32E8825650D0A,0xFFB4278825650D0A,0x1720208825551C7F,0x1721208825555C7F,
    0xFF253B88255A5C7F,0xFF252E8825595C7F,0xFF25278825535C7F,0xFF251D8825595C7F,
    0xFF25278825655C7F,0xFF251D8825655C7F,0x1728208825555D7F,0xFF263B88255A6E7F,
    0xFF262E8825596E7F,0xFF26278825536E7F,0xFF253B8825785C7F,0xFF252E8825655C7F,
    0xFF261D8825596E7F,0x172F2088255B457F,0xFF253B88255A457F,0xFF252E882559457F,
    0xFF253B882578457F,0xFF252E882565457F,0xFF2527882565457F,0xFF251D882565457F,
    0x173620882555117F,0xFF2527882553457F,0xFF251D882559457F,0xFF253B8825780D7F,
    0xFF252E8825650D7F,0xFF25278825650D7F,0xFF251D8825650D7F,0x173D208825556F7F,
    0xFF263B8825786E7F,0xFF262E8825656E7F,0xFF26278825656E7F,0xFF2627882553787F,
    0xFF261D882559787F,0xFF261D8825656E7F,0x17442088255B787F,0xFF263B88255A787F,
    0xFF262E882559787F,0xFF263B882578787F,0xFF262E882565787F,0xFF2627882565787F,
    0xFF261D882565787F,0x174B778825551C1C,0x174C7C8825551C38,0x174D7C88255B5C56,
    0xFF04848825595C56,0xFFB2798825655C56,0xFF048788255A5C56,0xFFB5808825655C56,
    0xFFB2848825655C56,0xFF04878825785C56,0x17547C8825554656,0xFF04798825595C56,
    0xFF04808825535C56,0xFFB3798825654556,0xFFB5808825654556,0xFFB3848825654556,
    0xFFB9878825784556,0x175B7C8825546F38,0xFF9B798825596E56,0xFF9B808825536E56,
    0xFF9B848825596E56,0xFF9B8788255A6E56,0xFF20798825597837,0xFF20808825537837,
    0x17627C8825551156,0xFF01798825594556,0xFF01808825534556,0xFF01848825594556,
    0xFF018788255A4556,0xFFB3848825650D56,0xFFB9878825780D56,0x17697C88255B7837,
    0xFF20848825597837,0xFFB2798825657837,0xFF208788255A7837,0xFFB5808825657837,
    0xFFB2848825657837,0xFF20878825787837,0x17707C8825550D56,0xFF81798825590D56,
    0xFF81808825530D56,0xFF81848825590D56,0xFFB3798825650D56,0xFF818788255A0D56,
    0xFFB5808825650D56,0x1777778825551C57,0x17787C8825556F56,0xFF1D798825597856,
    0xFFB2798825656E56,0xFFB5808825656E56,0xFF1D808825537856,0xFFB2848825656E56,
    0xFF1B878825786E56,0x177F7C88255B7856,0xFF1D848825597856,0xFFB2798825657856,
    0xFF1D8788255A7856,0xFFB5808825657856,0xFFB2848825657856,0xFF1D878825787856,
    0x17867788255B5C70,0xFF017388255A5C70,0xFF01848825595C70,0xFF04798825655C70,
    0xFF018788255A5C70,0xFFB6808825655C70,0xFF04848825655C70,0x178D7C8825554670,
    0xFF01798825595C70,0xFF01808825535C70,0xFF24798825654570,0xFFB6808825654570,
    0xFF24848825654570,0xFF24878825784570,0x17947C8825551170,0xFF23798825594570,
    0xFF23808825534570,0xFF23848825594570,0xFF238788255A4570,0xFF24848825650D70,
    0xFF24878825780D70,0x179B7C8825550D70,0xFF23798825590D70,0xFF23808825530D70,
    0xFF23848825590D70,0xFF24798825650D70,0xFF238788255A0D70,0xFFB6808825650D70,
    0x17A27C8825551C10,0x17A37C88255B5C37,0xFF04848825595C37,0xFFB2798825655C37,
    0xFF048788255A5C37,0xFFB5808825655C37,0xFFB2848825655C37,0xFFB1878825785C37,
    0x17AA7C8825554637,0xFF04798825595C37,0xFF04808825535C37,0xFFB3798825654537,
    0xFFB5808825654537,0xFFB3848825654537,0xFFB1878825784537,0x17B17C8825556E37,
    0xFF05798825596E37,0xFF05808825536E37,0xFF05848825596E37,0xFFB2798825656E37,
    0xFF058788255A6E37,0xFFB5808825656E37,0x17B87C8825551137,0xFF01798825594537,
    0xFF01808825534537,0xFF01848825594537,0xFF018788255A4537,0xFFB3848825650D37,
    0xFFB9878825780D37,0x17BF7C8825656F10,0xFFB2848825656E37,0xFFB1878825786E37,
    0xFFB279882565780A,0xFFB580882565780A,0xFFB284882565780A,0xFFB287882578780A,
    0x17C67C8825550D37,0xFF01798825590D37,0xFF01808825530D37,0xFF01848825590D37,
    0xFFB3798825650D37,0xFF018788255A0D37,0xFFB5808825650D37,0x17CD778825551C72,
    0x17CE778825554772,0xFF17798825596E70,0xFF17808825536E70,0xFF04738825785C70,
    0xFFA57388255A457F,0xFFA584882559457F,0xFF04878825785C70,0x17D57788255B6E70,
    0xFF177388255A6E70,0xFF17848825596E70,0xFF04798825656E70,0xFF178788255A6E70,
    0xFFB6808825656E70,0xFF04848825656E70,0x17DC778825556F70,0xFFA87388255A7870,
    0xFFA8798825597870,0xFF04738825786E70,0xFFA8808825537870,0xFFA8848825597870,
    0xFF04878825786E70,0x17E377882555117F,0xFFA579882559457F,0xFFA580882553457F,
    0xFFA5738825780D7F,0xFFA5808825650D7F,0xFFA5848825650D7F,0xFFA5878825780D7F,
    0x17EA7788255B7870,0xFF15738825787870,0xFF15798825657870,0xFFA88788255A7870,
    0xFFB6808825657870,0xFF15848825657870,0xFF15878825787870,0x17F1778825550D7F,
    0xFFA57388255A0D7F,0xFFA5798825590D7F,0xFFA5808825530D7F,0xFFA5848825590D7F,
    0xFFA5798825650D7F,0xFFA58788255A0D7F,0x17F87C8825551C0A,0x17F97C88255B5C0A,
    0xFFB2848825595C0A,0xFFB2798825655C0A,0xFFB28788255A5C0A,0xFFB5808825655C0A,
    0xFFB2848825655C0A,0xFFB2878825785C0A,0x18007C882555460A,0xFFB2798825595C0A,
    0xFFB2808825535C0A,0xFFB379882565450A,0xFFB580882565450A,0xFFB384882565450A,
    0xFFB387882578450A,0x18077C8825556E0A,0xFFB2798825596E0A,0xFFB2808825536E0A,
    0xFFB2848825596E0A,0xFFB2798825656E0A,0xFFB28788255A6E0A,0xFFB5808825656E0A,
    0x180E7C882555110A,0xFFB379882559450A,0xFFB380882553450A,0xFFB384882559450A,
    0xFFB38788255A450A,0xFFB3848825650D0A,0xFFB3878825780D0A,0x18157C8825556F0A,
    0xFFB279882559780A,0xFFB280882553780A,0xFFB284882559780A,0xFFB2848825656E0A,
    0xFFB28788255A780A,0xFFB2878825786E0A,0x181C7C8825550D0A,0xFFB3798825590D0A,
    0xFFB3808825530D0A,0xFFB3848825590D0A,0xFFB3798825650D0A,0xFFB38788255A0D0A,
    0xFFB5808825650D0A,0x182377882555487F,0x1824778825555C7F,0xFF257388255A5C7F,
    0xFF25798825595C7F,0xFF25808825535C7F,0xFF25848825595C7F,0xFF25798825655C7F,
    0xFF258788255A5C7F,0x182B778825555D7F,0xFF26798825596E7F,0xFF26808825536E7F,
    0xFF25738825785C7F,0xFF25808825655C7F,0xFF25848825655C7F,0xFF25878825785C7F,
    0x18327788255B6E7F,0xFF267388255A6E7F,0xFF26848825596E7F,0xFF26798825656E7F,
    0xFF268788255A6E7F,0xFF26808825656E7F,0xFF26848825656E7F,0x18397788255B457F,
    0xFFA573882578457F,0xFFA579882565457F,0xFFA58788255A457F,0xFFA580882565457F,
    0xFFA584882565457F,0xFFA587882578457F,0x1840778825556F7F,0xFF267388255A787F,
    0xFF2679882559787F,0xFF26738825786E7F,0xFF2680882553787F,0xFF2684882559787F,
    0xFF26878825786E7F,0x18477788255B787F,0xFF2673882578787F,0xFF2679882565787F,
    0xFF268788255A787F,0xFF2680882565787F,0xFF2684882565787F,0xFF2687882578787F,
    0x184E1C630F1C1C1C,0x184F2B610F551C1C,0x18502B610F551C1C,0x185131610F551C1C,
    0xFF1949610F521C1C,0xFF193B610F521C1C,0xFF015F610C654556,0xFF0549610C656E37,
    0xFFB23B61075B6E37,0xFFB42E6107656E37,0x18582A610C554556,0xFF0149610C594556,
    0xFF0149610C654556,0xFFB32E6107594556,0xFFB33B61075B4556,0xFFB42E6107654556,
    0xFFB4276107554556,0x185F2B610C551156,0xFF012E610C524556,0xFF01276125534556,
    0xFF815F610C650D56,0xFF8449610C650D56,0xFFB4276125650D56,0xFFB42E6107650D56,
    0x18662B610C556F37,0xFF055F610C656E37,0xFFA049610C597837,0xFFB4276125656E37,
    0xFFA02E610C527837,0xFFA0276125537837,0xFFB22E6107597837,0x186D2B610C557837,
    0xFF205F610C657837,0xFFA049610C657837,0xFFB23B61075B7837,0xFFB4276125657837,
    0xFFB42E6107657837,0xFFB4276107557837,0x18742A610C550D56,0xFF8449610C590D56,
    0xFF842E610C520D56,0xFF84276125530D56,0xFFB32E6107590D56,0xFFB33B61075B0D56,
    0xFFB4276107550D56,0x187B2B610C551C3A,0x187C31610C555D56,0xFF045F610C655C56,
    0xFF0449610C655C56,0xFF1B2E610C526E56,0xFFB22E6107596E56,0xFFB23B61075B5C56,
    0xFFB42E6107655C56,0x18832A610C554656,0xFF0449610C595C56,0xFF042E610C525C56,
    0xFF04276125535C56,0xFFB22E6107595C56,0xFFB4276107555C56,0xFFB4276125654556,
    0x188A2B610C556E56,0xFF1B49610C596E56,0xFF9B5F610C656E56,0xFF1B49610C656E56,
    0xFFB23B61075B6E56,0xFFB42E6107656E56,0xFFB4276107556E56,0x18912A610C557856,
    0xFF1D49610C597856,0xFF1D2E610C527856,0xFFB22E6107597856,0xFFB23B61075B7856,
    0xFFB42E6107657856,0xFFB4276107557856,0x189831610C551C3A,0xFF075F610C591C3A,
    0xFF075F610C521C3A,0xFF1D5F610C657856,0xFF1D49610C657856,0xFF232E610C520D70,
    0xFF242E6107590D70,0x189F2B610C550D70,0xFF2349610C590D70,0xFF235F610C650D70,
    0xFF2349610C650D70,0xFF243B61075B0D70,0xFFB62E6107650D70,0xFFB6276107550D70,
    0x18A62B610C551C10,0x18A72B610C555C37,0xFF045F610C655C37,0xFF0449610C655C37,
    0xFFB23B61075B5C37,0xFFB4276125655C37,0xFFB42E6107655C37,0xFFB4276107555C37,
    0x18AE2B610C554637,0xFF0449610C595C37,0xFF042E610C525C37,0xFF04276125535C37,
    0xFFB22E6107595C37,0xFF015F610C654537,0xFFB4276125654537,0x18B52A610C554537,
    0xFF0149610C594537,0xFF0149610C654537,0xFFB32E6107594537,0xFFB33B61075B4537,
    0xFFB42E6107654537,0xFFB4276107554537,0x18BC2B610C551137,0xFF012E610C524537,
    0xFF01276125534537,0xFF015F610C650D37,0xFF0149610C650D37,0xFFB4276125650D37,
    0xFFB42E6107650D37,0x18C32B610C556F10,0xFF0549610C596E37,0xFF052E610C526E37,
    0xFF05276125536E37,0xFFB22E6107596E37,0xFFB4276107556E37,0xFF055F610C65780A,
    0x18CA2A610C550D37,0xFF0149610C590D37,0xFF012E610C520D37,0xFF01276125530D37,
    0xFFB32E6107590D37,0xFFB33B61075B0D37,0xFFB4276107550D37,0x18D12B610C551C72,
    0x18D22A610C555C70,0xFF0449610C595C70,0xFF042E610C525C70,0xFF042E6107595C70,
    0xFF043B61075B5C70,0xFFB62E6107655C70,0xFFB6276107555C70,0x18D92B610C555D70,
    0xFF1549610C596E70,0xFF015F610C655C70,0xFF0449610C655C70,0xFF152E610C526E70,
    0xFF042E6107596E70,0xFFB6276107556E70,0x18E02B610C554570,0xFF2349610C594570,
    0xFF235F610C654570,0xFF2349610C654570,0xFF243B61075B4570,0xFFB62E6107654570,
    0xFFB6276107554570,0x18E731610C556F70,0xFF175F610C656E70,0xFF1549610C656E70,
    0xFF043B61075B6E70,0xFF152E610C527870,0xFF152E6107597870,0xFFB62E6107656E70,
    0x18EE2A610C551172,0xFF232E610C524570,0xFF242E6107594570,0xFF2549610C590D7F,
    0xFF022E610C520D7F,0xFF252E6107590D7F,0xFF25276107550D7F,0x18F52B610C557870,
    0xFF1549610C597870,0xFFA85F610C657870,0xFF1549610C657870,0xFF153B61075B7870,
    0xFFB62E6107657870,0xFFB6276107557870,0x18FC31610C551C0A,0x18FD5F610C551C0A,
    0xFF0B5F610C591C0A,0xFF0B5F610C521C0A,0xFF1E5F610C655C0A,0xFF1E5F610C656E0A,
    0xFF0C5F610C65450A,0xFF8C5F610C650D0A,0x190449610C5B5E0A,0xFF0C49610C595C0A,
    0xFF1E49610C596E0A,0xFFB249610C655C0A,0xFF0549610C59780A,0xFFB249610C656E0A,
    0xFFB249610C65780A,0x190B3C610C5B1C0A,0xFF0C49610C59450A,0xFF0C49610C590D0A,
    0xFFB349610C65450A,0xFFB23B61075B6E0A,0xFFB349610C650D0A,0xFFB23B61075B780A,
    0x19122F610C551C0A,0xFFB23B61075B5C0A,0xFFB33B61075B450A,0xFF052E610C52780A,
    0xFFB22E610759780A,0xFFB33B61075B0D0A,0xFFB42E610765780A,0x19192E610C555D0A,
    0xFF0C2E610C525C0A,0xFFB22E6107595C0A,0xFF1E2E610C526E0A,0xFFB22E6107596E0A,
    0xFFB42E6107655C0A,0xFFB42E6107656E0A,0x19202E610C55110A,0xFF0C2E610C52450A,
    0xFFB32E610759450A,0xFF0C2E610C520D0A,0xFFB32E6107590D0A,0xFFB42E610765450A,
    0xFFB42E6107650D0A,0x19272B610C551C7F,0x19282A610C555C7F,0xFF2549610C595C7F,
    0xFF022E610C525C7F,0xFF252E6107595C7F,0xFF253B61075B5C7F,0xFF252E6107655C7F,
    0xFF25276107555C7F,0x192F2B610C555D7F,0xFF2649610C596E7F,0xFF255F610C655C7F,
    0xFF2549610C655C7F,0xFF022E610C526E7F,0xFF262E6107596E7F,0xFF26276107556E7F,
    0x19362B610C55457F,0xFF2549610C59457F,0xFFA55F610C65457F,0xFF2549610C65457F,
    0xFF253B61075B457F,0xFF252E610765457F,0xFF2527610755457F,0x193D31610C55117F,
    0xFF022E610C52457F,0xFF252E610759457F,0xFFA55F610C650D7F,0xFF2549610C650D7F,
    0xFF253B61075B0D7F,0xFF252E6107650D7F,0x194431610C556F7F,0xFF265F610C656E7F,
    0xFF2649610C656E7F,0xFF263B61075B6E7F,0xFF022E610C52787F,0xFF262E610759787F,
    0xFF262E6107656E7F,0x194B2B610C55787F,0xFF2649610C59787F,0xFF265F610C65787F,
    0xFF2649610C65787F,0xFF263B61075B787F,0xFF262E610765787F,0xFF2627610755787F,
    0x19526C610F551C1C,0x19536C610F551C1C,0x19546C610F551C1C,0xFF1966610F521C1C,
    0xFF1973610F521C1C,0xFF2066610C657837,0xFFB27361075B7837,0xFFB5796107657837,
    0xFFB5846107657837,0x195B6C610C550D56,0xFF8166610C590D56,0xFF8179610C520D56,
    0xFFB3796107590D56,0xFF8184610C520D56,0xFFB3846107590D56,0xFFB5806107550D56,
    0x19626C610C555C37,0xFF0466610C595C37,0xFFB2796107595C37,0xFFB27361075B5C37,
    0xFFB2846107595C37,0xFFB5796107655C37,0xFFB5806107555C37,0x19696C610C555D37,
    0xFF0466610C655C37,0xFF0579610C526E37,0xFFB2796107596E37,0xFF0584610C526E37,
    0xFFB2846107596E37,0xFFB5846107655C37,0x19706C610C556E37,0xFF0566610C596E37,
    0xFF0566610C656E37,0xFFB27361075B6E37,0xFFB5796107656E37,0xFFB5806107556E37,
    0xFFB5846107656E37,0x19776C610C557837,0xFF2066610C597837,0xFF2079610C527837,
    0xFFB2796107597837,0xFF2084610C527837,0xFFB2846107597837,0xFFB5806107557837,
    0x197E6C610C551C56,0x197F6C610C555C56,0xFF0466610C595C56,0xFF0466610C655C56,
    0xFFB27361075B5C56,0xFFB5796107655C56,0xFFB5806107555C56,0xFFB5846107655C56,
    0x19866C610C554656,0xFF0479610C525C56,0xFFB2796107595C56,0xFF0166610C654556,
    0xFF0484610C525C56,0xFFB2846107595C56,0xFFB5846107654556,0x198D6B610C556E56,
    0xFF9B66610C596E56,0xFF9B79610C526E56,0xFFB2796107596E56,0xFFB27361075B6E56,
    0xFFB5796107656E56,0xFFB5806107556E56,0x19946C610C554556,0xFF0166610C594556,
    0xFFB3796107594556,0xFFB37361075B4556,0xFFB3846107594556,0xFFB5796107654556,
    0xFFB5806107554556,0x199B6B610C556F56,0xFF1D66610C597856,0xFF9B66610C656E56,
    0xFF1D79610C527856,0xFFB2796107597856,0xFFB27361075B7856,0xFFB5806107557856,
    0x19A26C610C551156,0xFF0179610C524556,0xFF0184610C524556,0xFF8166610C650D56,
    0xFFB37361075B0D56,0xFFB5796107650D56,0xFFB5846107650D56,0x19A96B610C551C57,
    0x19AA6A610C555D70,0xFF1766610C596E70,0xFF0166610C655C70,0xFF1779610C526E70,
    0xFF04796107596E70,0xFF047361075B5C70,0xFFB6796107655C70,0x19B16B610C554670,
    0xFF0166610C595C70,0xFF0179610C525C70,0xFF04796107595C70,0xFF2366610C654570,
    0xFFB6806107555C70,0xFFB6796107654570,0x19B86B610C551170,0xFF2366610C594570,
    0xFF2379610C524570,0xFF24796107594570,0xFF247361075B4570,0xFF2366610C650D70,
    0xFFB6806107554570,0x19BF6B610C556F70,0xFF1766610C656E70,0xFF047361075B6E70,
    0xFFA879610C527870,0xFF15796107597870,0xFFB6796107656E70,0xFFB6806107556E70,
    0x19C66B610C557857,0xFFA866610C597870,0xFF1D66610C657856,0xFFB5796107657856,
    0xFF157361075B7870,0xFFB6796107657870,0xFFB6806107557870,0x19CD6B610C550D70,
    0xFF2366610C590D70,0xFF2379610C520D70,0xFF24796107590D70,0xFF247361075B0D70,
    0xFFB6796107650D70,0xFFB6806107550D70,0x19D46C610C551C10,0x19D56C610C554710,
    0xFF0479610C525C37,0xFF0166610C654537,0xFF0484610C525C37,0xFFB27361075B6E0A,
    0xFFB5846107654537,0xFFB5796107656E0A,0x19DC6C610C554537,0xFF0166610C594537,
    0xFFB3796107594537,0xFFB37361075B4537,0xFFB3846107594537,0xFFB5796107654537,
    0xFFB5806107554537,0x19E36C610C551137,0xFF0179610C524537,0xFF0184610C524537,
    0xFF0166610C650D37,0xFFB37361075B0D37,0xFFB5796107650D37,0xFFB5846107650D37,
    0x19EA6C610C556F0A,0xFFB266610C656E0A,0xFF0579610C52780A,0xFFB279610759780A,
    0xFF0584610C52780A,0xFFB284610759780A,0xFFB5846107656E0A,0x19F16C610C55780A,
    0xFF0566610C59780A,0xFFB266610C65780A,0xFFB27361075B780A,0xFFB579610765780A,
    0xFFB580610755780A,0xFFB584610765780A,0x19F86C610C550D37,0xFF0166610C590D37,
    0xFF0179610C520D37,0xFFB3796107590D37,0xFF0184610C520D37,0xFFB3846107590D37,
    0xFFB5806107550D37,0x19FF6B610C551C72,0x1A006A610C555D7F,0xFF2666610C596E7F,
    0xFF2566610C655C7F,0xFF0279610C526E7F,0xFF26796107596E7F,0xFF257361075B5C7F,
    0xFF25796107655C7F,0x1A076B610C55467F,0xFF2566610C595C7F,0xFF0279610C525C7F,
    0xFF25796107595C7F,0xFFA566610C65457F,0xFF25806107555C7F,0xFFA579610765457F,
    0x1A0E6B610C55117F,0xFFA566610C59457F,0xFF0279610C52457F,0xFFA579610759457F,
    0xFFA57361075B457F,0xFFA566610C650D7F,0xFFA580610755457F,0x1A156B610C556F7F,
    0xFF2666610C656E7F,0xFF267361075B6E7F,0xFF0279610C52787F,0xFF2679610759787F,
    0xFF26796107656E7F,0xFF26806107556E7F,0x1A1C6B610C557872,0xFF2666610C59787F,
    0xFFA866610C657870,0xFF2666610C65787F,0xFF267361075B787F,0xFF2679610765787F,
    0xFF2680610755787F,0x1A236B610C550D7F,0xFFA566610C590D7F,0xFF0279610C520D7F,
    0xFFA5796107590D7F,0xFFA57361075B0D7F,0xFFA5796107650D7F,0xFFA5806107550D7F,
    0x1A2A6C610C55170A,0x1A2B6C610C59170A,0xFF0C66610C59450A,0xFF8C66610C590D0A,
    0xFFB2846107595C0A,0xFFB2846107596E0A,0xFFB384610759450A,0xFFB3846107590D0A,
    0x1A327B610C53170A,0xFFB2796107595C0A,0xFFB2796107596E0A,0xFFB379610759450A,
    0xFF1E84610C525C0A,0xFF1E84610C526E0A,0xFFB3796107590D0A,0x1A397B610C52170A,
    0xFF1E79610C525C0A,0xFF1E79610C526E0A,0xFF0C79610C52450A,0xFF0C84610C52450A,
    0xFF8C79610C520D0A,0xFF8C84610C520D0A,0x1A406B610C55170A,0xFF1E66610C595C0A,
    0xFF1E66610C596E0A,0xFFB5806107555C0A,0xFFB5806107556E0A,0xFFB580610755450A,
    0xFFB5806107550D0A,0x1A477461075B140A,0xFFB27361075B5C0A,0xFFB5796107655C0A,
    0xFFB37361075B450A,0xFFB579610765450A,0xFFB37361075B0D0A,0xFFB5796107650D0A,
    0x1A4E6C610C65140A,0xFFB266610C655C0A,0xFFB366610C65450A,0xFFB366610C650D0A,
    0xFFB5846107655C0A,0xFFB584610765450A,0xFFB5846107650D0A,0x1A5509610F551C1C,
    0x1A5608610F551C1C,0x1A5708610F551C1C,0xFF1906610F521C1C,0xFFA00661255A7837,
    0xFFB21D6125657837,0xFFA0066125787837,0xFFB41D6107657837,0xFFB20661075B7837,
    0x1A5E08610C551156,0xFF011D6125594556,0xFF010661255A4556,0xFF011D610C524556,
    0xFFB31D6107594556,0xFFB31D6125650D56,0xFFB1066125780D56,0x1A6508610C550D56,
    0xFF841D6125590D56,0xFF840661255A0D56,0xFF841D610C520D56,0xFFB31D6107590D56,
    0xFFB41D6107650D56,0xFFB30661075B0D56,0x1A6C08610C5B5C37,0xFF041D6125595C37,
    0xFF040661255A5C37,0xFFB21D6107595C37,0xFFB21D6125655C37,0xFFB41D6107655C37,
    0xFFB20661075B5C37,0x1A7308610C555D37,0xFF051D6125596E37,0xFF050661255A6E37,
    0xFF051D610C526E37,0xFFB21D6107596E37,0xFF04066125785C37,0xFFB20661075B6E37,
    0x1A7A08610C556F37,0xFFA01D6125597837,0xFFB21D6125656E37,0xFFA01D610C527837,
    0xFF05066125786E37,0xFFB21D6107597837,0xFFB41D6107656E37,0x1A8108610C551C57,
    0x1A8208610C555C56,0xFF041D6125595C56,0xFF040661255A5C56,0xFF041D610C525C56,
    0xFFB21D6107595C56,0xFFB41D6107655C56,0xFFB20661075B5C56,0x1A8908610C555D56,
    0xFF1B1D6125596E56,0xFF1B0661255A6E56,0xFF1B1D610C526E56,0xFFB21D6125655C56,
    0xFFB21D6107596E56,0xFFB1066125785C56,0x1A9008610C556F56,0xFFB21D6125656E56,
    0xFF1D1D610C527856,0xFFB1066125786E56,0xFFB21D6107597856,0xFFB41D6107656E56,
    0xFFB20661075B6E56,0x1A9708610C5B1157,0xFFB31D6125654556,0xFFB1066125784556,
    0xFFB41D6107654556,0xFFB30661075B4556,0xFF241D6125650D70,0xFF24066125780D70,
    0x1A9E08610C5B7856,0xFF1D1D6125597856,0xFF1D0661255A7856,0xFFB21D6125657856,
    0xFF1D066125787856,0xFFB41D6107657856,0xFFB20661075B7856,0x1AA508610C550D70,
    0xFF231D6125590D70,0xFF230661255A0D70,0xFF231D610C520D70,0xFF241D6107590D70,
    0xFFB61D6107650D70,0xFF240661075B0D70,0x1AAC09610C551C10,0x1AAD08610C554710,
    0xFF041D610C525C37,0xFFB21D6125596E0A,0xFFB31D6125654537,0xFFB1066125784537,
    0xFFB41D6107654537,0xFFB30661075B4537,0x1AB409610C556E0A,0xFFB20661255A6E0A,
    0xFFB4276125656E0A,0xFFB21D6125656E0A,0xFFB4276107556E0A,0xFFB41D6107656E0A,
    0xFFB20661075B6E0A,0x1ABB09610C556F0A,0xFFB227612553780A,0xFFB21D612559780A,
    0xFFB20661255A780A,0xFF051D610C52780A,0xFFB2066125786E0A,0xFFB21D610759780A,
    0x1AC208610C551137,0xFF011D6125594537,0xFF010661255A4537,0xFF011D610C524537,
    0xFFB31D6107594537,0xFFB31D6125650D37,0xFFB1066125780D37,0x1AC909610C55780A,
    0xFFB427612565780A,0xFFB21D612565780A,0xFFB427610755780A,0xFFB206612578780A,
    0xFFB41D610765780A,0xFFB20661075B780A,0x1AD008610C550D37,0xFF011D6125590D37,
    0xFF010661255A0D37,0xFF011D610C520D37,0xFFB31D6107590D37,0xFFB41D6107650D37,
    0xFFB30661075B0D37,0x1AD708610C551C72,0x1AD808610C555C70,0xFF041D6125595C70,
    0xFF040661255A5C70,0xFF041D610C525C70,0xFF041D6107595C70,0xFFB61D6107655C70,
    0xFF040661075B5C70,0x1ADF08610C555D70,0xFF151D6125596E70,0xFF150661255A6E70,
    0xFF151D610C526E70,0xFF041D6125655C70,0xFF041D6107596E70,0xFF04066125785C70,
    0x1AE608610C5B4570,0xFF231D6125594570,0xFF230661255A4570,0xFF241D6125654570,
    0xFF24066125784570,0xFFB61D6107654570,0xFF240661075B4570,0x1AED08610C541172,
    0xFF231D610C524570,0xFF241D6107594570,0xFF251D6125590D7F,0xFF250661255A0D7F,
    0xFF021D610C520D7F,0xFF251D6107590D7F,0x1AF408610C556F70,0xFF041D6125656E70,
    0xFF151D610C527870,0xFF04066125786E70,0xFF151D6107597870,0xFFB61D6107656E70,
    0xFF040661075B6E70,0x1AFB08610C5B7870,0xFF151D6125597870,0xFF150661255A7870,
    0xFF151D6125657870,0xFF15066125787870,0xFFB61D6107657870,0xFF150661075B7870,
    0x1B0209610C55170A,0x1B031E610C53170A,0xFFB327612553450A,0xFF0C1D610C525C0A,
    0xFF1E1D610C526E0A,0xFF0C1D610C52450A,0xFFB3276125530D0A,0xFF0C1D610C520D0A,
    0x1B0A1E610C53170A,0xFFB2276125535C0A,0xFFB2276125536E0A,0xFFB21D6107595C0A,
    0xFFB21D6107596E0A,0xFFB31D610759450A,0xFFB31D6107590D0A,0x1B110861255A140A,
    0xFFB21D6125595C0A,0xFFB20661255A5C0A,0xFFB31D612559450A,0xFFB30661255A450A,
    0xFFB31D6125590D0A,0xFFB30661255A0D0A,0x1B1809610755140A,0xFFB4276107555C0A,
    0xFFB427610755450A,0xFFB20661075B5C0A,0xFFB30661075B450A,0xFFB4276107550D0A,
    0xFFB30661075B0D0A,0x1B1F09612565140A,0xFFB4276125655C0A,0xFFB427612565450A,
    0xFFB2066125785C0A,0xFFB306612578450A,0xFFB4276125650D0A,0xFFB3066125780D0A,
    0x1B261D610C65140A,0xFFB21D6125655C0A,0xFFB31D612565450A,0xFFB41D6107655C0A,
    0xFFB41D610765450A,0xFFB31D6125650D0A,0xFFB41D6107650D0A,0x1B2D08610C551C7F,
    0x1B2E08610C555C7F,0xFF251D6125595C7F,0xFF250661255A5C7F,0xFF021D610C525C7F,
    0xFF251D6107595C7F,0xFF251D6107655C7F,0xFF250661075B5C7F,0x1B3508610C555D7F,
    0xFF261D6125596E7F,0xFF260661255A6E7F,0xFF021D610C526E7F,0xFF251D6125655C7F,
    0xFF261D6107596E7F,0xFF25066125785C7F,0x1B3C08610C5B457F,0xFF251D612559457F,
    0xFF250661255A457F,0xFF251D612565457F,0xFF2506612578457F,0xFF251D610765457F,
    0xFF250661075B457F,0x1B4308610C55117F,0xFF021D610C52457F,0xFF251D610759457F,
    0xFF251D6125650D7F,0xFF25066125780D7F,0xFF251D6107650D7F,0xFF250661075B0D7F,
    0x1B4A08610C556F7F,0xFF261D6125656E7F,0xFF021D610C52787F,0xFF26066125786E7F,
    0xFF261D610759787F,0xFF261D6107656E7F,0xFF260661075B6E7F,0x1B5108610C5B787F,
    0xFF261D612559787F,0xFF260661255A787F,0xFF261D612565787F,0xFF2606612578787F,
    0xFF261D610765787F,0xFF260661075B787F,0x1B581C620F551C1C,0x1B5984610C551C58,
    0x1B5A84610C555C72,0xFF0184610C525C70,0xFF04846107595C70,0xFF0284610C525C7F,
    0xFF25846107595C7F,0xFFB6846107655C70,0xFF25846107655C7F,0x1B6184610C536E58,
    0xFF9B84610C526E56,0xFFB2846107596E56,0xFF1784610C526E70,0xFF04846107596E70,
    0xFF0284610C526E7F,0xFF26846107596E7F,0x1B6884610C554572,0xFF2384610C524570,
    0xFF24846107594570,0xFF0284610C52457F,0xFFA584610759457F,0xFFB6846107654570,
    0xFFA584610765457F,0x1B6F84610C556F58,0xFF1D84610C527856,0xFFB5846107656E56,
    0xFFA884610C527870,0xFFB6846107656E70,0xFF0284610C52787F,0xFF26846107656E7F,
    0x1B768461075B7858,0xFFB2846107597856,0xFF15846107597870,0xFFB5846107657856,
    0xFF2684610759787F,0xFFB6846107657870,0xFF2684610765787F,0x1B7D84610C550D72,
    0xFF2384610C520D70,0xFF24846107590D70,0xFF0284610C520D7F,0xFFA5846107590D7F,
    0xFFB6846107650D70,0xFFA5846107650D7F,0x1B841C620F551C1C,0x1B851C62075B1C10,
    0xFFB38761075B450A,0xFFB20688075B5C37,0xFFB38761075B0D0A,0xFFB20688075B6E37,
    0xFFB30688075B4537,0xFFB20688075B7837,0x1B8C87610F551C1C,0xFF1987610F521C1C,
    0xFFB28761075B5C56,0xFFB38761075B4556,0xFFB28761075B6E37,0xFFB38761075B0D56,
    0xFFB28761075B7837,0x1B938761075B1C57,0xFFB28761075B6E56,0xFF048761075B5C70,
    0xFF048761075B6E70,0xFF248761075B4570,0xFFB28761075B7856,0xFF248761075B0D70,
    0x1B9A8761075B1C10,0xFFB28761075B5C37,0xFFB38761075B4537,0xFFB28761075B5C0A,
    0xFFB28761075B6E0A,0xFFB38761075B0D37,0xFFB28761075B780A,0x1BA10688075B1C10,
    0xFFB20688075B5C0A,0xFFB20688075B6E0A,0xFFB30688075B450A,0xFFB30688075B0D37,
    0xFFB20688075B780A,0xFFB30688075B0D0A,0x1BA88761075B1C72,0xFF258761075B5C7F,
    0xFF268761075B6E7F,0xFFA58761075B457F,0xFF158761075B7870,0xFF268761075B787F,
    0xFFA58761075B0D7F,0x1BAF28880C551C1C,0x1BB028880C555E39,0xFF052E880C526E37,
    0xFFB22E8807596E37,0xFFA02E880C527837,0xFFB6278807555C70,0xFFB42E8807656E37,
    0xFFB6278807556E70,0x1BB72E880C554637,0xFF042E880C525C37,0xFFB22E8807595C37,
    0xFF012E880C524537,0xFFB32E8807594537,0xFFB42E8807655C37,0xFFB42E8807654537,
    0x1BBE2E880C551C10,0xFF012E880C520D37,0xFFB32E8807590D37,0xFF052E880C52780A,
    0xFFB22E880759780A,0xFFB42E8807650D37,0xFFB42E880765780A,0x1BC5278807551C72,
    0xFF25278807555C7F,0xFF26278807556E7F,0xFF2527880755457F,0xFFB6278807557870,
    0xFF2627880755787F,0xFF25278807550D7F,0x1BCC2E880C555D0A,0xFF0C2E880C525C0A,
    0xFFB22E8807595C0A,0xFF1E2E880C526E0A,0xFFB22E8807596E0A,0xFFB42E8807655C0A,
    0xFFB42E8807656E0A,0x1BD32E880C55110A,0xFF0C2E880C52450A,0xFFB32E880759450A,
    0xFF0C2E880C520D0A,0xFFB32E8807590D0A,0xFFB42E880765450A,0xFFB42E8807650D0A,
    0x1BDA1E880C551C1C,0x1BDB278807551C38,0xFFB4278807554556,0xFFB4278807555C37,
    0xFFB4278807556E37,0xFFB4278807554537,0xFFB4278807550D56,0xFFB4278807557837,
    0x1BE21E8807551C57,0xFFB4278807555C56,0xFFB4278807556E56,0xFFB6278807554570,
    0xFFB4278807557856,0xFFB61D8807656E70,0xFFB6278807550D70,0x1BE9278807551C10,
    0xFFB4278807555C0A,0xFFB4278807556E0A,0xFFB427880755450A,0xFFB4278807550D37,
    0xFFB427880755780A,0xFFB4278807550D0A,0x1BF01D880C551C72,0xFF151D880C527870,
    0xFF151D8807597870,0xFF021D880C520D7F,0xFF251D8807590D7F,0xFFB61D8807657870,
    0xFF251D8807650D7F,0x1BF71D880C55467F,0xFF021D880C525C7F,0xFF251D8807595C7F,
    0xFF021D880C52457F,0xFF251D880759457F,0xFF251D8807655C7F,0xFF251D880765457F,
    0x1BFE1D880C556F7F,0xFF021D880C526E7F,0xFF261D8807596E7F,0xFF021D880C52787F,
    0xFF261D880759787F,0xFF261D8807656E7F,0xFF261D880765787F,0x1C051D880C551C39,
    0x1C061D880C555D39,0xFF1B1D880C526E56,0xFF041D8807595C70,0xFF051D880C526E37,
    0xFFB41D8807655C56,0xFFB41D8807655C37,0xFFB61D8807655C70,0x1C0D1D880C554639,
    0xFF041D880C525C56,0xFFB21D8807595C56,0xFF041D880C525C37,0xFF041D880C525C70,
    0xFFB21D8807595C37,0xFFB61D8807654570,0x1C141D880C556E39,0xFFB21D8807596E56,
    0xFF151D880C526E70,0xFFB21D8807596E37,0xFF041D8807596E70,0xFFB41D8807656E56,
    0xFFB41D8807656E37,0x1C1B1D880C554539,0xFF011D880C524556,0xFFB31D8807594556,
    0xFF231D880C524570,0xFF241D8807594570,0xFFB41D8807654556,0xFFB41D8807654537,
    0x1C221D880C557838,0xFF1D1D880C527856,0xFFB21D8807597856,0xFFA01D880C527837,
    0xFFB21D8807597837,0xFFB41D8807657856,0xFFB41D8807657837,0x1C291D880C550D57,
    0xFF841D880C520D56,0xFFB31D8807590D56,0xFF231D880C520D70,0xFF241D8807590D70,
    0xFFB41D8807650D56,0xFFB61D8807650D70,0x1C3008880F551C1C,0x1C3108880F551C1C,
    0xFF1906880F521C1C,0xFF011D880C524537,0xFFB31D8807594537,0xFFB20688075B5C56,
    0xFFB30688075B4556,0xFFB30688075B0D56,0x1C380688075B1C57,0xFFB20688075B6E56,
    0xFF040688075B5C70,0xFF040688075B6E70,0xFF240688075B4570,0xFFB20688075B7856,
    0xFF240688075B0D70,0x1C3F1D880C551C10,0xFF011D880C520D37,0xFFB31D8807590D37,
    0xFF051D880C52780A,0xFFB21D880759780A,0xFFB41D8807650D37,0xFFB41D880765780A,
    0x1C460688075B1C72,0xFF250688075B5C7F,0xFF260688075B6E7F,0xFF250688075B457F,
    0xFF150688075B7870,0xFF260688075B787F,0xFF250688075B0D7F,0x1C4D1D880C555D0A,
    0xFF0C1D880C525C0A,0xFFB21D8807595C0A,0xFF1E1D880C526E0A,0xFFB21D8807596E0A,
    0xFFB41D8807655C0A,0xFFB41D8807656E0A,0x1C541D880C55110A,0xFF0C1D880C52450A,
    0xFFB31D880759450A,0xFF0C1D880C520D0A,0xFFB31D8807590D0A,0xFFB41D880765450A,
    0xFFB41D8807650D0A,0x1C5B33880F551C1C,0x1C5C33880F551C1C,0x1C5D3E880F551C1C,
    0xFF1949880F521C1C,0xFF1966880F521C1C,0xFF193B880F521C1C,0xFF1973880F521C1C,
    0xFF205F880C657837,0xFF2066880C657837,0x1C6432880C555D56,0xFF045F880C655C56,
    0xFF0449880C655C56,0xFF0466880C655C56,0xFF1B2E880C526E56,0xFFB22E8807596E56,
    0xFFB42E8807655C56,0x1C6B32880C554656,0xFF0449880C595C56,0xFF0466880C595C56,
    0xFF042E880C525C56,0xFFB22E8807595C56,0xFF0166880C654556,0xFFB23B88075B5C56,
    0x1C7232880C5B4556,0xFF0149880C594556,0xFF0166880C594556,0xFF015F880C654556,
    0xFF0149880C654556,0xFFB33B88075B4556,0xFFB42E8807654556,0x1C7932880C551156,
    0xFF012E880C524556,0xFFB32E8807594556,0xFF815F880C650D56,0xFF8449880C650D56,
    0xFF8166880C650D56,0xFFB42E8807650D56,0x1C8033880C550D56,0xFF8449880C590D56,
    0xFF8166880C590D56,0xFF842E880C520D56,0xFFB32E8807590D56,0xFFB33B88075B0D56,
    0xFFB37388075B0D56,0x1C8732880C551C3A,0x1C8832880C5B6E56,0xFF1B49880C596E56,
    0xFF9B66880C596E56,0xFF9B5F880C656E56,0xFF1B49880C656E56,0xFFB23B88075B6E56,
    0xFFB42E8807656E56,0x1C8F32880C556F56,0xFF1D49880C597856,0xFF1D66880C597856,
    0xFF9B66880C656E56,0xFF1D2E880C527856,0xFFB22E8807597856,0xFFB23B88075B7856,
    0x1C9632880C551C3A,0xFF075F880C591C3A,0xFF075F880C521C3A,0xFF1D5F880C657856,
    0xFF1D49880C657856,0xFF1D66880C657856,0xFFB42E8807657856,0x1C9D32880C5B4570,
    0xFF2366880C594570,0xFF235F880C654570,0xFF2349880C654570,0xFF2366880C654570,
    0xFF243B88075B4570,0xFFB62E8807654570,0x1CA432880C551170,0xFF2349880C594570,
    0xFF232E880C524570,0xFF242E8807594570,0xFF235F880C650D70,0xFF2349880C650D70,
    0xFF2366880C650D70,0x1CAB32880C550D70,0xFF2349880C590D70,0xFF2366880C590D70,
    0xFF232E880C520D70,0xFF242E8807590D70,0xFF243B88075B0D70,0xFFB62E8807650D70,
    0x1CB233880C5B1C10,0x1CB33D880C5B5D37,0xFF0549880C596E37,0xFF0566880C596E37,
    0xFF045F880C655C37,0xFF0449880C655C37,0xFF0466880C655C37,0xFFB23B88075B6E37,
    0x1CBA3E880C5B4637,0xFF0449880C595C37,0xFF0466880C595C37,0xFF015F880C654537,
    0xFF0166880C654537,0xFFB23B88075B5C37,0xFFB27388075B5C37,0x1CC13E880C5B1137,
    0xFF0149880C594537,0xFF0166880C594537,0xFF0149880C654537,0xFFB33B88075B4537,
    0xFFB37388075B4537,0xFF0166880C650D37,0x1CC833880C5B6F37,0xFF055F880C656E37,
    0xFFA049880C597837,0xFF0549880C656E37,0xFF0566880C656E37,0xFFB27388075B6E37,
    0xFFB22E8807597837,0x1CCF33880C5B7810,0xFF2066880C597837,0xFFA049880C657837,
    0xFFB23B88075B7837,0xFFB27388075B7837,0xFFB266880C65780A,0xFFB42E8807657837,
    0x1CD63E880C5B0D37,0xFF0149880C590D37,0xFF0166880C590D37,0xFF015F880C650D37,
    0xFF0149880C650D37,0xFFB33B88075B0D37,0xFFB37388075B0D37,0x1CDD32880C551C72,
    0x1CDE32880C5B5C70,0xFF0166880C595C70,0xFF015F880C655C70,0xFF0449880C655C70,
    0xFF0166880C655C70,0xFF043B88075B5C70,0xFFB62E8807655C70,0x1CE532880C556E70,
    0xFF1549880C596E70,0xFF1766880C596E70,0xFF152E880C526E70,0xFF042E8807596E70,
    0xFF043B88075B6E70,0xFFB62E8807656E70,0x1CEC32880C551472,0xFF0449880C595C70,
    0xFF042E880C525C70,0xFF042E8807595C70,0xFFA55F880C650D7F,0xFF2549880C650D7F,
    0xFFA566880C650D7F,0x1CF332880C556F70,0xFF175F880C656E70,0xFF1549880C597870,
    0xFF1549880C656E70,0xFF1766880C656E70,0xFF152E880C527870,0xFF152E8807597870,
    0x1CFA32880C5B7870,0xFFA866880C597870,0xFFA85F880C657870,0xFF1549880C657870,
    0xFFA866880C657870,0xFF153B88075B7870,0xFFB62E8807657870,0x1D0132880C550D7F,
    0xFF2549880C590D7F,0xFFA566880C590D7F,0xFF022E880C520D7F,0xFF252E8807590D7F,
    0xFF253B88075B0D7F,0xFF252E8807650D7F,0x1D083E880C551C0A,0x1D094B880C551C0A,
    0xFF0B5F880C591C0A,0xFF0B5F880C521C0A,0xFF0C49880C595C0A,0xFF1E66880C595C0A,
    0xFF0C5F880C65450A,0xFFB366880C65450A,0x1D103E880C5B5D0A,0xFF1E49880C596E0A,
    0xFF1E5F880C655C0A,0xFFB249880C655C0A,0xFFB266880C655C0A,0xFFB23B88075B5C0A,
    0xFFB27388075B5C0A,0x1D173E880C5B6E0A,0xFF1E66880C596E0A,0xFF1E5F880C656E0A,
    0xFFB249880C656E0A,0xFFB266880C656E0A,0xFFB23B88075B6E0A,0xFFB27388075B6E0A,
    0x1D1E3E880C5B110A,0xFF0C49880C59450A,0xFF0C66880C59450A,0xFFB349880C65450A,
    0xFFB33B88075B450A,0xFFB37388075B450A,0xFFB366880C650D0A,0x1D253E880C5B780A,
    0xFF0549880C59780A,0xFF0566880C59780A,0xFF055F880C65780A,0xFFB249880C65780A,
    0xFFB23B88075B780A,0xFFB27388075B780A,0x1D2C3E880C5B0D0A,0xFF0C49880C590D0A,
    0xFF8C66880C590D0A,0xFF8C5F880C650D0A,0xFFB349880C650D0A,0xFFB33B88075B0D0A,
    0xFFB37388075B0D0A,0x1D3332880C55487F,0x1D3430880C59487F,0xFF2549880C595C7F,
    0xFF2649880C596E7F,0xFF2549880C59457F,0xFF2649880C59787F,0xFF262E8807596E7F,
    0xFF262E880759787F,0x1D3B2E880C53487F,0xFF022E880C525C7F,0xFF252E8807595C7F,
    0xFF022E880C526E7F,0xFF022E880C52457F,0xFF252E880759457F,0xFF022E880C52787F,
    0x1D423D880C5B487F,0xFF2566880C595C7F,0xFF2666880C596E7F,0xFFA566880C59457F,
    0xFF2666880C59787F,0xFF253B88075B5C7F,0xFF253B88075B457F,0x1D492F88075B487F,
    0xFF263B88075B6E7F,0xFF252E8807655C7F,0xFF262E8807656E7F,0xFF252E880765457F,
    0xFF263B88075B787F,0xFF262E880765787F,0x1D504A880C65487F,0xFF255F880C655C7F,
    0xFF2549880C655C7F,0xFFA55F880C65457F,0xFF2649880C656E7F,0xFF2549880C65457F,
    0xFF2649880C65787F,0x1D5760880C65487F,0xFF2566880C655C7F,0xFF265F880C656E7F,
    0xFF2666880C656E7F,0xFFA566880C65457F,0xFF265F880C65787F,0xFF2666880C65787F,
    0x1D5E1C890F1C1C1C,0x1D5F1C890F1C1C1C,0x1D601C890F1C1C1C,0xFFB71C8A0C1C1C1C,
    0xFF1987880F521C1C,0xFF8179880C520D56,0xFF8184880C520D56,0xFFB5798807657837,
    0xFFB5848807657837,0x1D67778807555C56,0xFFB27388075B5C56,0xFFB2848807595C56,
    0xFFB5798807655C56,0xFFB5808807555C56,0xFFB5848807655C56,0xFFB28788075B5C56,
    0x1D6E7C880C554656,0xFF0479880C525C56,0xFFB2798807595C56,0xFF0484880C525C56,
    0xFFB5798807654556,0xFFB5848807654556,0xFFB38788075B4556,0x1D7576880C554556,
    0xFF0179880C524556,0xFFB3798807594556,0xFFB37388075B4556,0xFF0184880C524556,
    0xFFB3848807594556,0xFFB5808807554556,0x1D7C7C8807550D56,0xFFB3798807590D56,
    0xFFB3848807590D56,0xFFB5798807650D56,0xFFB5808807550D56,0xFFB5848807650D56,
    0xFFB38788075B0D56,0x1D837C880C557837,0xFF2079880C527837,0xFFB2798807597837,
    0xFF2084880C527837,0xFFB2848807597837,0xFFB5808807557837,0xFFB28788075B7837,
    0x1D8A77880C551C57,0x1D8B76880C556E56,0xFF9B79880C526E56,0xFFB2798807596E56,
    0xFFB27388075B6E56,0xFF9B84880C526E56,0xFFB2848807596E56,0xFFB5808807556E56,
    0x1D927C880C556F56,0xFF1D79880C527856,0xFFB2798807597856,0xFFB5798807656E56,
    0xFF1D84880C527856,0xFFB5848807656E56,0xFFB28788075B6E56,0x1D99778807557856,
    0xFFB27388075B7856,0xFFB2848807597856,0xFFB5798807657856,0xFFB5808807557856,
    0xFFB5848807657856,0xFFB28788075B7856,0x1DA0778807554570,0xFF247388075B4570,
    0xFF24848807594570,0xFFB6798807654570,0xFFB6808807554570,0xFFB6848807654570,
    0xFF248788075B4570,0x1DA77C880C551170,0xFF2379880C524570,0xFF24798807594570,
    0xFF2384880C524570,0xFFB6798807650D70,0xFFB6848807650D70,0xFF248788075B0D70,
    0x1DAE76880C550D70,0xFF2379880C520D70,0xFF24798807590D70,0xFF247388075B0D70,
    0xFF2384880C520D70,0xFF24848807590D70,0xFFB6808807550D70,0x1DB57C880C551C10,
    0x1DB67C8807555C37,0xFFB2798807595C37,0xFFB2848807595C37,0xFFB5798807655C37,
    0xFFB5808807555C37,0xFFB5848807655C37,0xFFB28788075B5C37,0x1DBD7C880C554637,
    0xFF0479880C525C37,0xFF0484880C525C37,0xFFB5798807654537,0xFFB5808807554537,
    0xFFB5848807654537,0xFFB38788075B4537,0x1DC47C880C556E37,0xFF0579880C526E37,
    0xFFB2798807596E37,0xFF0584880C526E37,0xFFB2848807596E37,0xFFB5808807556E37,
    0xFFB28788075B6E37,0x1DCB7B880C551137,0xFF0179880C524537,0xFFB3798807594537,
    0xFF0184880C524537,0xFFB3848807594537,0xFFB5798807650D37,0xFFB5848807650D37,
    0x1DD27C8807556F10,0xFFB5798807656E37,0xFFB5848807656E37,0xFFB579880765780A,
    0xFFB580880755780A,0xFFB584880765780A,0xFFB28788075B780A,0x1DD97C880C550D37,
    0xFF0179880C520D37,0xFFB3798807590D37,0xFF0184880C520D37,0xFFB3848807590D37,
    0xFFB5808807550D37,0xFFB38788075B0D37,0x1DE077880C551C72,0x1DE1778807555C70,
    0xFF047388075B5C70,0xFF04848807595C70,0xFFB6798807655C70,0xFFB6808807555C70,
    0xFFB6848807655C70,0xFF048788075B5C70,0x1DE876880C556E70,0xFF1779880C526E70,
    0xFF04798807596E70,0xFF047388075B6E70,0xFF1784880C526E70,0xFF04848807596E70,
    0xFFB6808807556E70,0x1DEF7C880C551472,0xFF0179880C525C70,0xFF04798807595C70,
    0xFF0184880C525C70,0xFFA5798807650D7F,0xFFA5848807650D7F,0xFFA58788075B0D7F,
    0x1DF67C880C556F70,0xFFA879880C527870,0xFF15798807597870,0xFFB6798807656E70,
    0xFFA884880C527870,0xFFB6848807656E70,0xFF048788075B6E70,0x1DFD778807557870,
    0xFF157388075B7870,0xFF15848807597870,0xFFB6798807657870,0xFFB6808807557870,
    0xFFB6848807657870,0xFF158788075B7870,0x1E0476880C550D7F,0xFF0279880C520D7F,
    0xFFA5798807590D7F,0xFFA57388075B0D7F,0xFF0284880C520D7F,0xFFA5848807590D7F,
    0xFFA5808807550D7F,0x1E0B7C880C551C0A,0x1E0C7C8807555C0A,0xFFB2798807595C0A,
    0xFFB2848807595C0A,0xFFB5798807655C0A,0xFFB5808807555C0A,0xFFB5848807655C0A,
    0xFFB28788075B5C0A,0x1E137C880C55460A,0xFF1E79880C525C0A,0xFF1E84880C525C0A,
    0xFFB579880765450A,0xFFB580880755450A,0xFFB584880765450A,0xFFB38788075B450A,
    0x1E1A7C880C556E0A,0xFF1E79880C526E0A,0xFFB2798807596E0A,0xFF1E84880C526E0A,
    0xFFB2848807596E0A,0xFFB5808807556E0A,0xFFB28788075B6E0A,0x1E217B880C55110A,
    0xFF0C79880C52450A,0xFFB379880759450A,0xFF0C84880C52450A,0xFFB384880759450A,
    0xFFB5798807650D0A,0xFFB5848807650D0A,0x1E287B880C556F0A,0xFF0579880C52780A,
    0xFFB279880759780A,0xFFB5798807656E0A,0xFF0584880C52780A,0xFFB284880759780A,
    0xFFB5848807656E0A,0x1E2F7C880C550D0A,0xFF8C79880C520D0A,0xFFB3798807590D0A,
    0xFF8C84880C520D0A,0xFFB3848807590D0A,0xFFB5808807550D0A,0xFFB38788075B0D0A,
    0x1E3677880C55487F,0x1E37778807555C7F,0xFF257388075B5C7F,0xFF25848807595C7F,
    0xFF25798807655C7F,0xFF25808807555C7F,0xFF25848807655C7F,0xFF258788075B5C7F,
    0x1E3E7C880C55467F,0xFF0279880C525C7F,0xFF25798807595C7F,0xFF0284880C525C7F,
    0xFFA579880765457F,0xFFA584880765457F,0xFFA58788075B457F,0x1E4576880C556E7F,
    0xFF0279880C526E7F,0xFF26798807596E7F,0xFF267388075B6E7F,0xFF0284880C526E7F,
    0xFF26848807596E7F,0xFF26808807556E7F,0x1E4C76880C55457F,0xFF0279880C52457F,
    0xFFA579880759457F,0xFFA57388075B457F,0xFF0284880C52457F,0xFFA584880759457F,
    0xFFA580880755457F,0x1E537C880C556F7F,0xFF0279880C52787F,0xFF2679880759787F,
    0xFF26798807656E7F,0xFF0284880C52787F,0xFF26848807656E7F,0xFF268788075B6E7F,
    0x1E5A77880755787F,0xFF267388075B787F,0xFF2684880759787F,0xFF2679880765787F,
    0xFF2680880755787F,0xFF2684880765787F,0xFF268788075B787F,0x1E611C627E551C1C,
    0x1E6218617F551C1C,0x1E6316617F551C38,0x1E644B617F521C38,0xFF1066617F525C37,
    0xFF1049617F526E37,0xFF1066617F526E37,0xFF1066617F524537,0xFF1049617F520D56,
    0xFF1049617F527837,0x1E6B3C617F521C38,0xFF1049617F525C37,0xFF103B617F525C37,
    0xFF1049617F524537,0xFF103B617F526E37,0xFF103B617F520D56,0xFF103B617F527837,
    0x1E720E617F521C38,0xFF103B617F524537,0xFF1006617F525C37,0xFF1006617F526E37,
    0xFF1006617F524537,0xFF1006617F520D56,0xFF1006617F527837,0x1E7921617F554837,
    0xFF1066617F527837,0xFF041D617F555C37,0xFF0127617F554537,0xFF051D617F556E37,
    0xFF011D617F554537,0xFFA01D617F557837,0x1E8016617F551C38,0xFF065F617F5B1C38,
    0xFF0649617F5B1C38,0xFF0666617F5B1C38,0xFF063B617F5B1C38,0xFF0606617F5B1C38,
    0xFFA02E617F557837,0x1E8728617F554837,0xFF042E617F555C37,0xFF0427617F555C37,
    0xFF052E617F556E37,0xFF012E617F554537,0xFF0527617F556E37,0xFFA027617F557837,
    0x1E8E16617F551C3A,0x1E8F60617F521C3A,0xFF1066617F525C56,0xFF075F617F521C3A,
    0xFF1066617F526E56,0xFF1066617F524556,0xFF1066617F527856,0xFF1066617F520D56,
    0x1E963C617F521C57,0xFF1049617F525C56,0xFF1049617F526E56,0xFF1049617F524556,
    0xFF1049617F527856,0xFF1049617F520D70,0xFF103B617F520D70,0x1E9D2F617F554856,
    0xFF103B617F525C56,0xFF103B617F526E56,0xFF103B617F524556,0xFF1B2E617F556E56,
    0xFF103B617F527856,0xFF1D2E617F557856,0x1EA428617F551C56,0xFF042E617F555C56,
    0xFF0427617F555C56,0xFF012E617F554556,0xFF1B27617F556E56,0xFF842E617F550D56,
    0xFF1D27617F557856,0x1EAB1E617F551C56,0xFF041D617F555C56,0xFF0127617F554556,
    0xFF1B1D617F556E56,0xFF011D617F554556,0xFF8427617F550D56,0xFF1D1D617F557856,
    0x1EB208617F551C57,0xFF1006617F525C56,0xFF1006617F526E56,0xFF1006617F524556,
    0xFF1006617F527856,0xFF841D617F550D56,0xFF1006617F520D70,0x1EB918617F551C10,
    0x1EBA33617F551C10,0xFF0B5F617F5B1C0A,0xFF0C49617F5B5C0A,0xFF1E66617F5B5C0A,
    0xFF0C3B617F5B5C0A,0xFF1E73617F5B5C0A,0xFF012E617F550D37,0x1EC118617F556E0A,
    0xFF1A49617F526E0A,0xFF1A66617F526E0A,0xFF1A3B617F526E0A,0xFF1E73617F526E0A,
    0xFF1A06617F526E0A,0xFF1E1D617F556E0A,0x1EC816617F556E0A,0xFF1E49617F5B6E0A,
    0xFF1E66617F5B6E0A,0xFF1E3B617F5B6E0A,0xFF1E2E617F556E0A,0xFF1E27617F556E0A,
    0xFF1E06617F5B6E0A,0x1ECF18617F556F0A,0xFF1A49617F52780A,0xFF1A66617F52780A,
    0xFF1E73617F5B6E0A,0xFF1A3B617F52780A,0xFF1A06617F52780A,0xFF051D617F55780A,
    0x1ED616617F55780A,0xFF0549617F5B780A,0xFF0566617F5B780A,0xFF053B617F5B780A,
    0xFF052E617F55780A,0xFF0527617F55780A,0xFF0506617F5B780A,0x1EDD16617F550D37,
    0xFF1049617F520D37,0xFF1066617F520D37,0xFF103B617F520D37,0xFF0127617F550D37,
    0xFF1006617F520D37,0xFF011D617F550D37,0x1EE416617F551C72,0x1EE521617F555C70,
    0xFF1049617F525C70,0xFF1066617F525C70,0xFF103B617F525C70,0xFF042E617F555C70,
    0xFF0427617F555C70,0xFF041D617F555C70,0x1EEC16617F554670,0xFF1049617F524570,
    0xFF1066617F524570,0xFF232E617F554570,0xFF1006617F525C70,0xFF2327617F554570,
    0xFF231D617F554570,0x1EF316617F556E70,0xFF1049617F526E70,0xFF1066617F526E70,
    0xFF103B617F526E70,0xFF1527617F556E70,0xFF1006617F526E70,0xFF151D617F556E70,
    0x1EFA16617F551170,0xFF103B617F524570,0xFF1066617F520D70,0xFF1006617F524570,
    0xFF232E617F550D70,0xFF2327617F550D70,0xFF231D617F550D70,0x1F0116617F556F70,
    0xFF1049617F527870,0xFF1066617F527870,0xFF152E617F556E70,0xFF103B617F527870,
    0xFF1006617F527870,0xFF151D617F557870,0x1F0813617F551C72,0xFFB85F617F5B1C72,
    0xFFB849617F5B1C72,0xFFB83B617F5B1C72,0xFF152E617F557870,0xFFB806617F5B1C72,
    0xFF1527617F557870,0x1F0F16617F551C72,0x1F1016617F551C72,0xFFB866617F5B1C72,
    0xFF0249617F520D7F,0xFF0266617F520D7F,0xFF023B617F520D7F,0xFF0206617F520D7F,
    0xFF021D617F550D7F,0x1F1716617F555D7F,0xFF0266617F525C7F,0xFF023B617F526E7F,
    0xFF022E617F555C7F,0xFF0227617F555C7F,0xFF021D617F555C7F,0xFF0206617F526E7F,
    0x1F1E12617F55467F,0xFF0249617F525C7F,0xFF023B617F525C7F,0xFF022E617F55457F,
    0xFF0206617F525C7F,0xFF0227617F55457F,0xFF021D617F55457F,0x1F2516617F55117F,
    0xFF0249617F52457F,0xFF0266617F52457F,0xFF023B617F52457F,0xFF0206617F52457F,
    0xFF022E617F550D7F,0xFF0227617F550D7F,0x1F2C16617F556F7F,0xFF0249617F526E7F,
    0xFF0266617F526E7F,0xFF022E617F556E7F,0xFF0227617F556E7F,0xFF021D617F556E7F,
    0xFF0206617F52787F,0x1F3321617F55787F,0xFF0249617F52787F,0xFF0266617F52787F,
    0xFF023B617F52787F,0xFF022E617F55787F,0xFF0227617F55787F,0xFF021D617F55787F,
    0x1F3A18617F551C0A,0x1F3B4B617F551C0A,0xFF0B5F617F521C0A,0xFF1A49617F525C0A,
    0xFF1A66617F52450A,0xFF0C66617F5B450A,0xFF1A66617F520D0A,0xFF8C66617F5B0D0A,
    0x1F423C617F55140A,0xFF1A3B617F525C0A,0xFF1A49617F52450A,0xFF0C49617F5B450A,
    0xFF1A49617F520D0A,0xFF0C3B617F5B450A,0xFF0C49617F5B0D0A,0x1F4967617F55140A,
    0xFF1A66617F525C0A,0xFF1E73617F525C0A,0xFF0C73617F52450A,0xFF0C73617F5B450A,
    0xFF8C73617F520D0A,0xFF8C73617F5B0D0A,0x1F502F617F55140A,0xFF1A3B617F52450A,
    0xFF0C2E617F555C0A,0xFF0C2E617F55450A,0xFF1A3B617F520D0A,0xFF0C3B617F5B0D0A,
    0xFF0C2E617F550D0A,0x1F571E617F55140A,0xFF0C27617F555C0A,0xFF0C1D617F555C0A,
    0xFF0C27617F55450A,0xFF0C1D617F55450A,0xFF0C27617F550D0A,0xFF0C1D617F550D0A,
    0x1F5E06617F55140A,0xFF1A06617F525C0A,0xFF1A06617F52450A,0xFF0C06617F5B5C0A,
    0xFF0C06617F5B450A,0xFF1A06617F520D0A,0xFF0C06617F5B0D0A,0x1F651C627F551C1C,
    0x1F662C627F551C1C,0x1F6773617F521C57,0xFF0473617F525C56,0xFF9B73617F526E56,
    0xFF0173617F524556,0xFF1D73617F527856,0xFF8173617F520D56,0xFF2373617F520D70,
    0x1F6E73617F521C10,0xFF0473617F525C37,0xFF0573617F526E37,0xFF0173617F524537,
    0xFF2073617F527837,0xFF0173617F520D37,0xFF0573617F52780A,0x1F7573617F521C72,
    0xFF0173617F525C70,0xFF1773617F526E70,0xFF2373617F524570,0xFF0273617F52457F,
    0xFFA873617F527870,0xFF0273617F520D7F,0x1F7C33627F551C1C,0xFF0673617F5B1C38,
    0xFFB873617F5B1C72,0xFF0573617F5B780A,0xFF0C2E887F555C0A,0xFF0C2E887F55450A,
    0xFF0C2E887F550D0A,0x1F832C627F551772,0xFF0273617F525C7F,0xFF0273617F526E7F,
    0xFF0427887F555C70,0xFF1527887F556E70,0xFF2327887F554570,0xFF2327887F550D70,
    0x1F8A27887F551C72,0xFF0227887F555C7F,0xFF0227887F556E7F,0xFF0227887F55457F,
    0xFF1527887F557870,0xFF0227887F55787F,0xFF0227887F550D7F,0x1F9175617F551C1C,
    0x1F927A617F551C38,0xFF0479617F555C56,0xFF9B79617F556E56,0xFF0179617F554556,
    0xFF8179617F550D56,0xFF2079617F557837,0xFF2080617F557837,0x1F9979617F551C57,
    0xFF0179617F555C70,0xFF1779617F556E70,0xFF2379617F554570,0xFF1D79617F557856,
    0xFFA879617F557870,0xFF2379617F550D70,0x1FA07A617F554737,0xFF0479617F555C37,
    0xFF0480617F555C37,0xFF0579617F556E37,0xFF0179617F554537,0xFF0580617F556E37,
    0xFF0180617F554537,0x1FA77A617F551C10,0xFF1E79617F556E0A,0xFF0179617F550D37,
    0xFF1E80617F556E0A,0xFF0180617F550D37,0xFF0579617F55780A,0xFF0580617F55780A,
    0x1FAE7A617F55140A,0xFF1E79617F555C0A,0xFF1E80617F555C0A,0xFF0C79617F55450A,
    0xFF0C80617F55450A,0xFF8C79617F550D0A,0xFF8C80617F550D0A,0x1FB574617F551C7F,
    0xFF0279617F555C7F,0xFF0279617F556E7F,0xFF0279617F55457F,0xFF0273617F52787F,
    0xFF0279617F55787F,0xFF0279617F550D7F,0x1FBC81617F551C1C,0x1FBD81617F554756,
    0xFF0480617F555C56,0xFF0484617F555C56,0xFF9B80617F556E56,0xFF0180617F554556,
    0xFF9B84617F556E56,0xFF0184617F554556,0x1FC481617F551C38,0xFF0484617F555C37,
    0xFF0584617F556E37,0xFF8180617F550D56,0xFF0184617F554537,0xFF8184617F550D56,
    0xFF2084617F557837,0x1FCB81617F551C57,0xFF2380617F554570,0xFF1D80617F557856,
    0xFF2384617F554570,0xFF1D84617F557856,0xFF2380617F550D70,0xFF2384617F550D70,
    0x1FD281617F555E70,0xFF0180617F555C70,0xFF0184617F555C70,0xFF1780617F556E70,
    0xFF1784617F556E70,0xFFA880617F557870,0xFFA884617F557870,0x1FD984617F551C10,
    0xFF1E84617F555C0A,0xFF1E84617F556E0A,0xFF0C84617F55450A,0xFF0184617F550D37,
    0xFF0584617F55780A,0xFF8C84617F550D0A,0x1FE081617F551C7F,0xFF0280617F555C7F,
    0xFF0280617F556E7F,0xFF0280617F55457F,0xFF0280617F55787F,0xFF0280617F550D7F,
    0xFF0284617F550D7F,0x1FE71E887F551C1C,0x1FE81E887F554756,0xFF0427887F555C56,
    0xFF041D887F555C56,0xFF1B27887F556E56,0xFF0127887F554556,0xFF1B1D887F556E56,
    0xFF011D887F554556,0x1FEF1E887F551C38,0xFF0527887F556E37,0xFF051D887F556E37,
    0xFF8427887F550D56,0xFF841D887F550D56,0xFFA027887F557837,0xFFA01D887F557837,
    0x1FF61E887F551C57,0xFF041D887F555C70,0xFF1D27887F557856,0xFF151D887F556E70,
    0xFF231D887F554570,0xFF1D1D887F557856,0xFF231D887F550D70,0x1FFD1E887F551437,
    0xFF0427887F555C37,0xFF041D887F555C37,0xFF0127887F554537,0xFF011D887F554537,
    0xFF0127887F550D37,0xFF011D887F550D37,0x20041D887F551C72,0xFF021D887F555C7F,
    0xFF021D887F556E7F,0xFF021D887F55457F,0xFF151D887F557870,0xFF021D887F55787F,
    0xFF021D887F550D7F,0x200B1E887F551C0A,0xFF0C27887F555C0A,0xFF1E27887F556E0A,
    0xFF0C27887F55450A,0xFF0527887F55780A,0xFF0C27887F550D0A,0xFF051D887F55780A,
    0x201285617F551C1C,0x201387617F551C38,0xFF0187617F524556,0xFF0487617F525C37,
    0xFF0687617F5B1C38,0xFF0587617F526E37,0xFF8187617F520D56,0xFF2087617F527837,
    0x201A87617F521C57,0xFF0487617F525C56,0xFF9B87617F526E56,0xFF0187617F525C70,
    0xFF2387617F524570,0xFF1D87617F527856,0xFF2387617F520D70,0x202187617F551C10,
    0xFF0187617F524537,0xFF1E87617F526E0A,0xFF0187617F520D37,0xFF1E87617F5B6E0A,
    0xFF0587617F52780A,0xFF0587617F5B780A,0x202885617F551C72,0xFF1787617F526E70,
    0xFFB887617F5B1C72,0xFF0287617F52457F,0xFFA887617F527870,0xFF0284617F55457F,
    0xFF0287617F520D7F,0x202F87617F55140A,0xFF1E87617F525C0A,0xFF0C87617F52450A,
    0xFF1E87617F5B5C0A,0xFF0C87617F5B450A,0xFF8C87617F520D0A,0xFF8C87617F5B0D0A,
    0x203685617F555E7F,0xFF0287617F525C7F,0xFF0284617F555C7F,0xFF0287617F526E7F,
    0xFF0284617F556E7F,0xFF0287617F52787F,0xFF0284617F55787F,0x203D08887F551C1C,
    0x203E06887F551C38,0xFF1006887F525C56,0xFF1006887F526E56,0xFF1006887F524556,
    0xFF0606887F5B1C38,0xFF1006887F520D56,0xFF1006887F527837,0x204506887F521C57,
    0xFF1006887F525C70,0xFF1006887F526E70,0xFF1006887F524570,0xFF1006887F527856,
    0xFF1006887F527870,0xFF1006887F520D70,0x204C06887F551C10,0xFF1006887F525C37,
    0xFF1006887F526E37,0xFF1006887F524537,0xFF1006887F520D37,0xFF1A06887F52780A,
    0xFF0506887F5B780A,0x205306887F551C72,0xFF0206887F525C7F,0xFFB806887F5B1C72,
    0xFF0206887F526E7F,0xFF0206887F52457F,0xFF0206887F52787F,0xFF0206887F520D7F,
    0x205A08887F555D0A,0xFF1A06887F525C0A,0xFF0C1D887F555C0A,0xFF1A06887F526E0A,
    0xFF0C06887F5B5C0A,0xFF1E1D887F556E0A,0xFF1E06887F5B6E0A,0x206108887F55110A,
    0xFF1A06887F52450A,0xFF0C1D887F55450A,0xFF0C06887F5B450A,0xFF1A06887F520D0A,
    0xFF0C1D887F550D0A,0xFF0C06887F5B0D0A,0x206832887E551C1C,0x206932887E551C38,
    0x206A3D887E551C38,0xFF1049887D525C56,0xFF065F887F5B1C38,0xFF0166887D654556,
    0xFF0649887F5B1C38,0xFF0666887F5B1C38,0xFF063B887F5B1C38,0x207132887E554556,
    0xFF0149887D594556,0xFFA366887D594556,0xFF1049887F524556,0xFF1066887F524556,
    0xFF0149887D654556,0xFF012E887F554556,0x207832887E551156,0xFF1049887D524556,
    0xFFA366887D524556,0xFF103B887F524556,0xFF8449887D650D56,0xFF8166887D650D56,
    0xFF842E887F550D56,0x207F32887E556F37,0xFF0566887D596E37,0xFF1049887F526E37,
    0xFF1066887F526E37,0xFF0566887D656E37,0xFF2066887D527837,0xFF052E887F556E37,
    0x208632887E557837,0xFF2066887D597837,0xFF1049887F527837,0xFF1066887F527837,
    0xFF2066887D657837,0xFF103B887F527837,0xFFA02E887F557837,0x208D3D887E530D56,
    0xFF8449887D590D56,0xFFA366887D590D56,0xFFA366887D520D56,0xFF1049887F520D56,
    0xFF1066887F520D56,0xFF103B887F520D56,0x209432887E551C3A,0x20953D887E535C56,
    0xFF0449887D595C56,0xFFA366887D595C56,0xFFA366887D525C56,0xFF1049887F525C56,
    0xFF1066887F525C56,0xFF103B887F525C56,0x209C32887E555D56,0xFF1049887D526E56,
    0xFFA366887D526E56,0xFF0449887D655C56,0xFF0466887D655C56,0xFF103B887F526E56,
    0xFF042E887F555C56,0x20A332887E556E56,0xFF1B49887D596E56,0xFFA366887D596E56,
    0xFF1049887F526E56,0xFF1066887F526E56,0xFF1B49887D656E56,0xFF1B2E887F556E56,
    0x20AA3D887E556F56,0xFF9B66887D656E56,0xFF1049887D527856,0xFF1D66887D527856,
    0xFF1049887F527856,0xFF1066887F527856,0xFF103B887F527856,0x20B132887E551C3A,
    0xFF075F887F521C3A,0xFF1D49887D597856,0xFF1D66887D597856,0xFF1D49887D657856,
    0xFF1D66887D657856,0xFF1D2E887F557856,0x20B83D887E530D70,0xFF2349887D590D70,
    0xFF1049887D520D70,0xFFA366887D520D70,0xFF1049887F520D70,0xFF1066887F520D70,
    0xFF103B887F520D70,0x20BF32887E551C70,0x20C04B887D591770,0xFF0449887D595C70,
    0xFFA366887D595C70,0xFF1549887D596E70,0xFF2349887D594570,0xFFA366887D594570,
    0xFFA366887D590D70,0x20C74B887D524870,0xFF1049887D525C70,0xFFA366887D525C70,
    0xFF1049887D526E70,0xFF1049887D524570,0xFFA366887D524570,0xFF1049887D527870,
    0x20CE3D887E524870,0xFFA366887D526E70,0xFF103B887F525C70,0xFF1049887F524570,
    0xFF103B887F526E70,0xFF103B887F524570,0xFF103B887F527870,0x20D54B887F524870,
    0xFF1049887F525C70,0xFF1066887F525C70,0xFF1049887F526E70,0xFF1066887F526E70,
    0xFF1066887F524570,0xFF1049887F527870,0x20DC32887E551770,0xFFA366887D596E70,
    0xFF042E887F555C70,0xFF2349887D650D70,0xFF152E887F556E70,0xFF232E887F554570,
    0xFF232E887F550D70,0x20E34B887D651770,0xFF0449887D655C70,0xFF0166887D655C70,
    0xFF1549887D656E70,0xFF2349887D654570,0xFF2366887D654570,0xFF2366887D650D70,
    0x20EA32887E551C10,0x20EB32887E555D10,0xFF0466887D595C37,0xFF0566887D526E37,
    0xFF0466887D655C37,0xFF103B887F526E37,0xFF042E887F555C37,0xFF1E49887F5B6E0A,
    0x20F232887E554637,0xFF0466887D525C37,0xFF1049887F525C37,0xFF1066887F525C37,
    0xFF103B887F525C37,0xFF0166887D654537,0xFF012E887F554537,0x20F93D887E551137,
    0xFF0166887D594537,0xFF0166887D524537,0xFF1049887F524537,0xFF1066887F524537,
    0xFF103B887F524537,0xFF0166887D650D37,0x21003D887E556F0A,0xFF1E66887D656E0A,
    0xFF0566887D52780A,0xFF1E66887F5B6E0A,0xFF1A49887F52780A,0xFF1A66887F52780A,
    0xFF1A3B887F52780A,0x210732887E55780A,0xFF0566887D59780A,0xFF0566887D65780A,
    0xFF0549887F5B780A,0xFF0566887F5B780A,0xFF053B887F5B780A,0xFF052E887F55780A,
    0x210E32887E550D37,0xFF0166887D590D37,0xFF0166887D520D37,0xFF1049887F520D37,
    0xFF1066887F520D37,0xFF103B887F520D37,0xFF012E887F550D37,0x211531887E551C72,
    0x211631887E551C72,0xFF1549887D597870,0xFFB85F887F5B1C72,0xFFB849887F5B1C72,
    0xFFB83B887F5B1C72,0xFF1549887D657870,0xFF152E887F557870,0x211D30887E555C7F,
    0xFF0249887D595C7F,0xFF0249887D525C7F,0xFF0249887F525C7F,0xFF0249887D655C7F,
    0xFF023B887F525C7F,0xFF022E887F555C7F,0x212430887E556E7F,0xFF0249887D596E7F,
    0xFF0249887D526E7F,0xFF0249887F526E7F,0xFF0249887D656E7F,0xFF023B887F526E7F,
    0xFF022E887F556E7F,0x212B30887E55457F,0xFF0249887D59457F,0xFF0249887D52457F,
    0xFF0249887F52457F,0xFF0249887D65457F,0xFF023B887F52457F,0xFF022E887F55457F,
    0x213230887E55787F,0xFF0249887D59787F,0xFF0249887D52787F,0xFF0249887F52787F,
    0xFF0249887D65787F,0xFF023B887F52787F,0xFF022E887F55787F,0x213930887E550D7F,
    0xFF0249887D590D7F,0xFF0249887D520D7F,0xFF0249887F520D7F,0xFF0249887D650D7F,
    0xFF023B887F520D7F,0xFF022E887F550D7F,0x214032887E551C0A,0x21414A887F551C0A,
    0xFF0B5F887F521C0A,0xFF1A49887F525C0A,0xFF0B5F887F5B1C0A,0xFF1A49887F526E0A,
    0xFF0C49887F5B5C0A,0xFF0C49887F5B450A,0x214866887E555D0A,0xFF1E66887D595C0A,
    0xFF1E66887D596E0A,0xFF1E66887D526E0A,0xFF1E66887D655C0A,0xFF1A66887F526E0A,
    0xFF1E66887F5B5C0A,0x214F66887E55460A,0xFF1E66887D525C0A,0xFF0C66887D59450A,
    0xFF1A66887F525C0A,0xFF1A66887F52450A,0xFF0C66887D65450A,0xFF0C66887F5B450A,
    0x215666887E55110A,0xFF0C66887D52450A,0xFF8C66887D590D0A,0xFF8C66887D520D0A,
    0xFF1A66887F520D0A,0xFF8C66887D650D0A,0xFF8C66887F5B0D0A,0x215D3C887F55170A,
    0xFF1A49887F52450A,0xFF1A3B887F526E0A,0xFF0C3B887F5B5C0A,0xFF1E3B887F5B6E0A,
    0xFF1A49887F520D0A,0xFF0C49887F5B0D0A,0x21642F887F55170A,0xFF1A3B887F525C0A,
    0xFF1A3B887F52450A,0xFF0C3B887F5B450A,0xFF1E2E887F556E0A,0xFF1A3B887F520D0A,
    0xFF0C3B887F5B0D0A,0x216B6B887E551C1C,0x216C74887E551C38,0x216D74887E555D56,
    0xFFA373887D526E56,0xFFA379887D536E56,0xFF0373887D655C56,0xFF9B73887F526E56,
    0xFF0379887D655C56,0xFF0479887F555C56,0x217474887E554656,0xFFA373887D595C56,
    0xFFA373887D525C56,0xFFA379887D535C56,0xFF0473887F525C56,0xFF0373887D654556,
    0xFF0379887D654556,0x217B74887E551156,0xFFA373887D594556,0xFFA373887D524556,
    0xFFA379887D534556,0xFF0173887F524556,0xFF0179887F554556,0xFF0379887D650D56,
    0x218274887E550D56,0xFFA373887D590D56,0xFFA373887D520D56,0xFFA379887D530D56,
    0xFF8173887F520D56,0xFF0373887D650D56,0xFF8179887F550D56,0x218974887E551C38,
    0xFF0673887F5B1C38,0xFF2073887D597837,0xFF2079887D537837,0xFF2273887D657837,
    0xFF2279887D657837,0xFF2079887F557837,0x219074887E556F37,0xFF0573887D596E37,
    0xFF2273887D656E37,0xFF2073887D527837,0xFF2279887D656E37,0xFF0579887F556E37,
    0xFF2073887F527837,0x219774887E551C57,0x219874887E555D70,0xFFA373887D526E70,
    0xFFA379887D536E70,0xFF0173887D655C70,0xFF1773887F526E70,0xFF0179887D655C70,
    0xFF0179887F555C70,0x219F74887E554670,0xFFA373887D595C70,0xFFA373887D525C70,
    0xFFA379887D535C70,0xFF0173887F525C70,0xFF2373887D654570,0xFF2379887D654570,
    0x21A674887E556F57,0xFFA373887D596E56,0xFFA373887D596E70,0xFF2273887D656E56,
    0xFF1D73887D527856,0xFF2279887D656E56,0xFF9B79887F556E56,0x21AD74887E551170,
    0xFFA373887D594570,0xFFA373887D524570,0xFFA379887D534570,0xFF2373887F524570,
    0xFF2379887F554570,0xFF2379887D650D70,0x21B474887E557856,0xFF1D73887D597856,
    0xFF1D79887D537856,0xFF1D73887F527856,0xFF2273887D657856,0xFF2279887D657856,
    0xFF1D79887F557856,0x21BB74887E550D70,0xFFA373887D590D70,0xFFA373887D520D70,
    0xFFA379887D530D70,0xFF2373887F520D70,0xFF2373887D650D70,0xFF2379887F550D70,
    0x21C274887E551C10,0x21C374887E555D37,0xFF0573887D526E37,0xFF0579887D536E37,
    0xFF0373887D655C37,0xFF0573887F526E37,0xFF0379887D655C37,0xFF0479887F555C37,
    0x21CA74887E554637,0xFF0473887D595C37,0xFF0473887D525C37,0xFF0479887D535C37,
    0xFF0473887F525C37,0xFF8373887D654537,0xFF8379887D654537,0x21D174887E551137,
    0xFF0173887D594537,0xFF0173887D524537,0xFF0179887D534537,0xFF0173887F524537,
    0xFF0179887F554537,0xFF8379887D650D37,0x21D874887E550D37,0xFF0173887D590D37,
    0xFF0173887D520D37,0xFF0179887D530D37,0xFF0173887F520D37,0xFF8373887D650D37,
    0xFF0179887F550D37,0x21DF74887E556F0A,0xFF2273887D656E0A,0xFF0573887D52780A,
    0xFF2279887D656E0A,0xFF1E73887F5B6E0A,0xFF1E79887F556E0A,0xFF0573887F52780A,
    0x21E674887E55780A,0xFF0573887D59780A,0xFF0579887D53780A,0xFF2273887D65780A,
    0xFF2279887D65780A,0xFF0573887F5B780A,0xFF0579887F55780A,0x21ED6A887E551C72,
    0x21EE6A887E551C72,0xFF1766887D656E70,0xFF1773887D656E70,0xFFB866887F5B1C72,
    0xFFB873887F5B1C72,0xFF1779887F556E70,0xFF0279887F55457F,0x21F56A887E53457F,
    0xFFA366887D59457F,0xFFA373887D59457F,0xFFA373887D52457F,0xFFA379887D53457F,
    0xFF0266887F52457F,0xFF0273887F52457F,0x21FC6A887E556F70,0xFFA866887D527870,
    0xFFA873887D527870,0xFF1779887D656E70,0xFFA879887D537870,0xFF1066887F527870,
    0xFFA873887F527870,0x22036A887E55117F,0xFFA366887D52457F,0xFFA373887D590D7F,
    0xFF0266887D650D7F,0xFF0273887D650D7F,0xFF0279887D650D7F,0xFF0279887F550D7F,
    0x220A6A887E557870,0xFFA866887D597870,0xFFA873887D597870,0xFFA866887D657870,
    0xFFA873887D657870,0xFFA879887D657870,0xFFA879887F557870,0x22116A887E530D7F,
    0xFFA366887D590D7F,0xFFA366887D520D7F,0xFFA373887D520D7F,0xFFA379887D530D7F,
    0xFF0266887F520D7F,0xFF0273887F520D7F,0x221875887E55170A,0x221975887D53140A,
    0xFF1E73887D595C0A,0xFF0C73887D59450A,0xFF0380887D535C0A,0xFF8380887D53450A,
    0xFF8C73887D590D0A,0xFF8380887D530D0A,0x222074887E53170A,0xFF1E79887D535C0A,
    0xFF1E73887F525C0A,0xFF1E79887D536E0A,0xFF0C79887D53450A,0xFF1E73887F526E0A,
    0xFF8C79887D530D0A,0x222773887E52170A,0xFF1E73887D525C0A,0xFF1E73887D526E0A,
    0xFF0C73887D52450A,0xFF0C73887F52450A,0xFF8C73887D520D0A,0xFF8C73887F520D0A,
    0x222E75887E55170A,0xFF1E73887D596E0A,0xFF1E79887F555C0A,0xFF0C79887F55450A,
    0xFF0C80887F55450A,0xFF8C79887F550D0A,0xFF8C80887F550D0A,0x223575887E55140A,
    0xFF1E73887F5B5C0A,0xFF8373887D65450A,0xFF0C73887F5B450A,0xFF1E80887F555C0A,
    0xFF8373887D650D0A,0xFF8C73887F5B0D0A,0x223C75887D65140A,0xFF0373887D655C0A,
    0xFF0379887D655C0A,0xFF8379887D65450A,0xFF8380887D65450A,0xFF8379887D650D0A,
    0xFF8380887D650D0A,0x22436A887E55487F,0x22446A887D535E7F,0xFFA366887D595C7F,
    0xFFA366887D596E7F,0xFFA379887D535C7F,0xFFA379887D536E7F,0xFF0266887D59787F,
    0xFF0279887D53787F,0x224B67887D525E7F,0xFFA366887D525C7F,0xFFA373887D525C7F,
    0xFFA366887D526E7F,0xFFA373887D526E7F,0xFF0266887D52787F,0xFF0273887D52787F,
    0x225267887F525E7F,0xFF0266887F525C7F,0xFF0273887F525C7F,0xFF0266887F526E7F,
    0xFF0273887F526E7F,0xFF0266887F52787F,0xFF0273887F52787F,0x225974887E555E7F,
    0xFFA373887D595C7F,0xFFA373887D596E7F,0xFF0273887D59787F,0xFF0279887F555C7F,
    0xFF0279887F556E7F,0xFF0279887F55787F,0x226067887D65487F,0xFF0266887D655C7F,
    0xFF0273887D655C7F,0xFF0266887D656E7F,0xFF0266887D65457F,0xFF0273887D65457F,
    0xFF0266887D65787F,0x226774887D65487F,0xFF0279887D655C7F,0xFF0273887D656E7F,
    0xFF0279887D656E7F,0xFF0279887D65457F,0xFF0273887D65787F,0xFF0279887D65787F,
    0x226E82887E551C1C,0x226F82887E554856,0x227085887D534856,0xFFA384887D535C56,
    0xFFA387887D595C56,0xFFA384887D536E56,0xFFA384887D534556,0xFFA387887D594556,
    0xFF1D84887D537856,0x227782887E534856,0xFFA380887D535C56,0xFFA380887D536E56,
    0xFFA380887D534556,0xFF2280887D537856,0xFF9B87887F526E56,0xFF1D87887F527856,
    0x227E87887E524856,0xFFA387887D525C56,0xFFA387887D526E56,0xFFA387887D524556,
    0xFF0487887F525C56,0xFF0187887F524556,0xFF1D87887D527856,0x228581887E554756,
    0xFF0380887D655C56,0xFF2280887D656E56,0xFF0380887D654556,0xFF0484887F555C56,
    0xFF9B84887F556E56,0xFF0184887F554556,0x228C82887E554856,0xFFA387887D596E56,
    0xFF0480887F555C56,0xFF9B80887F556E56,0xFF0180887F554556,0xFF1D87887D597856,
    0xFF1D80887F557856,0x229385887D654756,0xFF0384887D655C56,0xFF0387887D655C56,
    0xFF2284887D656E56,0xFF0384887D654556,0xFF2287887D656E56,0xFF0387887D654556,
    0x229A82887E551C38,0x229B82887E550D56,0xFFA387887D590D56,0xFF0380887D650D56,
    0xFF8180887F550D56,0xFF0384887D650D56,0xFF0387887D650D56,0xFF8184887F550D56,
    0x22A282887E551C38,0xFFA380887D530D56,0xFFA384887D530D56,0xFF0687887F5B1C38,
    0xFFA387887D520D56,0xFF8187887F520D56,0xFF2287887D657837,0x22A982887E555D37,
    0xFF0380887D655C37,0xFF0587887D526E37,0xFF0480887F555C37,0xFF0384887D655C37,
    0xFF0387887D655C37,0xFF0484887F555C37,0x22B082887E556E37,0xFF2280887D536E37,
    0xFF0584887D536E37,0xFF0587887D596E37,0xFF0580887F556E37,0xFF0587887F526E37,
    0xFF0584887F556E37,0x22B782887E556F37,0xFF2280887D656E37,0xFF2280887D537837,
    0xFF2284887D656E37,0xFF2287887D656E37,0xFF2087887D527837,0xFF2087887F527837,
    0x22BE82887E557837,0xFF2084887D537837,0xFF2087887D597837,0xFF2280887D657837,
    0xFF2080887F557837,0xFF2284887D657837,0xFF2084887F557837,0x22C582887E551C57,
    0x22C681887E555D70,0xFFA380887D536E70,0xFFA384887D536E70,0xFF0180887D655C70,
    0xFF0184887D655C70,0xFF1780887F556E70,0xFF1784887F556E70,0x22CD82887E554670,
    0xFFA380887D535C70,0xFFA384887D535C70,0xFF0180887F555C70,0xFF0184887F555C70,
    0xFF2384887D654570,0xFF2387887D654570,0x22D482887E554570,0xFFA380887D534570,
    0xFFA384887D534570,0xFFA387887D594570,0xFF2380887D654570,0xFF2380887F554570,
    0xFF2384887F554570,0x22DB82887E556F57,0xFF1780887D656E70,0xFF2280887D657856,
    0xFF1784887D656E70,0xFF2284887D657856,0xFF2287887D657856,0xFF1D84887F557856,
    0x22E282887E551170,0xFFA387887D524570,0xFF2387887F524570,0xFF2380887D650D70,
    0xFF2384887D650D70,0xFF2387887D650D70,0xFF2384887F550D70,0x22E982887E550D70,
    0xFFA380887D530D70,0xFFA384887D530D70,0xFFA387887D590D70,0xFFA387887D520D70,
    0xFF2380887F550D70,0xFF2387887F520D70,0x22F082887E551C10,0x22F182887E554637,
    0xFF0380887D535C37,0xFF0484887D535C37,0xFF0487887D525C37,0xFF0487887F525C37,
    0xFF8384887D654537,0xFF8387887D654537,0x22F882887E554537,0xFF8380887D534537,
    0xFF0184887D534537,0xFF0187887D594537,0xFF8380887D654537,0xFF0180887F554537,
    0xFF0184887F554537,0x22FF82887E535E10,0xFF0487887D595C37,0xFF2280887D53780A,
    0xFF0584887D53780A,0xFF0587887D59780A,0xFF0587887D52780A,0xFF0587887F52780A,
    0x230682887E551137,0xFF0187887D524537,0xFF0187887F524537,0xFF8380887D650D37,
    0xFF8384887D650D37,0xFF8387887D650D37,0xFF0184887F550D37,0x230D82887E55780A,
    0xFF2280887D65780A,0xFF0580887F55780A,0xFF2284887D65780A,0xFF2287887D65780A,
    0xFF0584887F55780A,0xFF0587887F5B780A,0x231482887E550D37,0xFF8380887D530D37,
    0xFF0184887D530D37,0xFF0187887D590D37,0xFF0187887D520D37,0xFF0180887F550D37,
    0xFF0187887F520D37,0x231B81887E551C72,0x231C81887E557870,0xFFA880887D537870,
    0xFFA884887D537870,0xFFA880887D657870,0xFFA880887F557870,0xFFA884887D657870,
    0xFFA884887F557870,0x232381887E555C7F,0xFFA380887D535C7F,0xFFA384887D535C7F,
    0xFF0280887D655C7F,0xFF0280887F555C7F,0xFF0284887D655C7F,0xFF0284887F555C7F,
    0x232A81887E556E7F,0xFFA380887D536E7F,0xFFA384887D536E7F,0xFF0280887D656E7F,
    0xFF0280887F556E7F,0xFF0284887D656E7F,0xFF0284887F556E7F,0x233181887E55457F,
    0xFFA380887D53457F,0xFFA384887D53457F,0xFF0280887D65457F,0xFF0280887F55457F,
    0xFF0284887D65457F,0xFF0284887F55457F,0x233881887E55787F,0xFF0280887D53787F,
    0xFF0284887D53787F,0xFF0280887D65787F,0xFF0280887F55787F,0xFF0284887D65787F,
    0xFF0284887F55787F,0x233F81887E550D7F,0xFFA380887D530D7F,0xFFA384887D530D7F,
    0xFF0280887D650D7F,0xFF0280887F550D7F,0xFF0284887D650D7F,0xFF0284887F550D7F,
    0x234682887E55170A,0x234782887E555C0A,0xFF1E84887D535C0A,0xFF1E87887D595C0A,
    0xFF0380887D655C0A,0xFF0384887D655C0A,0xFF1E84887F555C0A,0xFF1E87887F5B5C0A,
    0x234E82887E555D0A,0xFF2280887D536E0A,0xFF1E84887D536E0A,0xFF1E87887D596E0A,
    0xFF1E87887D526E0A,0xFF0387887D655C0A,0xFF1E87887F526E0A,0x235585887E55460A,
    0xFF1E87887D525C0A,0xFF1E87887F525C0A,0xFF8384887D65450A,0xFF8387887D65450A,
    0xFF0C84887F55450A,0xFF0C87887F5B450A,0x235C82887E556E0A,0xFF2280887D656E0A,
    0xFF1E80887F556E0A,0xFF2284887D656E0A,0xFF2287887D656E0A,0xFF1E84887F556E0A,
    0xFF1E87887F5B6E0A,0x236385887E55110A,0xFF0C84887D53450A,0xFF0C87887D59450A,
    0xFF0C87887D52450A,0xFF0C87887F52450A,0xFF8384887D650D0A,0xFF8387887D650D0A,
    0x236A85887E550D0A,0xFF8C84887D530D0A,0xFF8C87887D590D0A,0xFF8C87887D520D0A,
    0xFF8C87887F520D0A,0xFF8C84887F550D0A,0xFF8C87887F5B0D0A,0x237187887E551C72,
    0x237287887D555D70,0xFFA387887D595C70,0xFFA387887D525C70,0xFFA387887D596E70,
    0xFFA387887D526E70,0xFF0187887D655C70,0xFF1787887D656E70,0x237987887D551C72,
    0xFFA887887D597870,0xFFA887887D527870,0xFFA387887D590D7F,0xFFA887887D657870,
    0xFFA387887D520D7F,0xFF0287887D650D7F,0x238087887D55467F,0xFFA387887D595C7F,
    0xFFA387887D525C7F,0xFFA387887D59457F,0xFFA387887D52457F,0xFF0287887D655C7F,
    0xFF0287887D65457F,0x238787887D556F7F,0xFFA387887D596E7F,0xFFA387887D526E7F,
    0xFF0287887D59787F,0xFF0287887D656E7F,0xFF0287887D52787F,0xFF0287887D65787F,
    0x238E87887F551C72,0xFF0187887F525C70,0xFF1787887F526E70,0xFFB887887F5B1C72,
    0xFF0287887F52457F,0xFFA887887F527870,0xFF0287887F520D7F,0x239587887F525E7F,
    0xFF0287887F525C7F,0xFF0287887F526E7F,0xFF0287887F52787F,
};


typedef struct {
    int depth;
    int index;
} StackFrame;

typedef struct {
    StackFrame frames[128];
    size_t pointer;
} Stack;

__device__ __host__ StackFrame new_frame(int index, int depth) {
    StackFrame frame;
    frame.depth = depth;
    frame.index = index;
    return frame;
}

__device__ __host__ void push(Stack *stack, StackFrame frame) {
    stack->frames[stack->pointer] = frame;
    stack->pointer++;
}

__device__ __host__ StackFrame pop(Stack *stack) {
    stack->pointer--;
    return stack->frames[stack->pointer];
}

__device__ __host__ int get_resulting_node_improved(const uint64_t np[6], const BiomeTree *bt) {
    Stack stack;
    stack.pointer = 0;
    
    push(&stack, new_frame(0, 0));

    uint64_t best_dist = 18446744073709551615ULL;
    int best_index = -1;
    int visited_nodes = 0;

    while (stack.pointer > 0) {
        if (best_dist == 0) {
            break;
        }

        StackFrame frame = pop(&stack);

        uint64_t node = btree20_nodes[frame.index];
        visited_nodes++;

        // if the node is a leaf
        if (((node >> 8 * 7) & 0xFF) == 0xFF) {
            uint64_t dist = get_np_dist(np, bt, frame.index);

            if (dist < best_dist) {
                best_dist = dist;
                best_index = frame.index;
            }
        }

        // push each child node onto the stack
        int child_spacing = btree20_steps[frame.depth];
        
        if (child_spacing == 0) {
            continue;
        }

        uint64_t local_best = 18446744073709551615ULL;
        for (size_t i = 0, child_index = btree20_nodes[frame.index] >> 48; i < btree20_order && child_index < 9112; i++, child_index += child_spacing) {
            uint64_t child_score = get_np_dist(np, bt, child_index);

            if (child_score > local_best) {
                continue;
            }
            if (child_score > best_dist) {
                continue; //no need to search the children in this case
            }
            if (child_score < local_best) {
                local_best = child_score;
            }
            // if this child has a less score, there is potential to find a better leaf...
            push(&stack, new_frame(child_index, frame.depth+1));
        }
    }   
    // printf("visited nodes = %d -> %ld\n", visited_nodes, (btree20_nodes[best_index] >> 8 * 6) & 0xFF);
    // exit(1);
    return (btree20_nodes[best_index] >> 8 * 6) & 0xFF;
}

__device__ static const BiomeTree g_btree[MC_NEWEST - MC_1_18 + 1] =
{
    // MC_1_20
    { btree20_steps, &btree20_param[0][0], btree20_nodes, btree20_order,
        sizeof(btree20_nodes) / sizeof(uint64_t) },
};

__host__ __device__ int climateToBiome(int mc, const uint64_t np[6], uint64_t *dat)
{
    if (mc < MC_1_18 || mc > MC_NEWEST)
        return -1;

    const BiomeTree *bt = &g_btree[0];
    int alt = 0;
    int idx;
    uint64_t ds = -1;

    idx = get_resulting_node_improved(np, bt);
    return idx;
}

/// Biome sampler for MC 1.18
__host__ __device__ int sampleBiomeNoise(const BiomeNoise *bn, int64_t *np, int x, int y, int z,
    uint64_t *dat, uint32_t sample_flags)
{
    if (bn->nptype >= 0)
    {   // initialized for a specific climate parameter
        if (np)
            memset(np, 0, NP_MAX*sizeof(*np));
        int64_t id = (int64_t) (10000.0 * sampleClimatePara(bn, np, x, z));
        return (int) id;
    }

    float t = 0, h = 0, c = 0, e = 0, d = 0, w = 0;
    double px = x, pz = z;
    if (!(sample_flags & SAMPLE_NO_SHIFT))
    {
        px += sampleDoublePerlin(&bn->climate[NP_SHIFT], x, 0, z) * 4.0;
        pz += sampleDoublePerlin(&bn->climate[NP_SHIFT], z, x, 0) * 4.0;
    }

    c = sampleDoublePerlin(&bn->climate[NP_CONTINENTALNESS], px, 0, pz);
    e = sampleDoublePerlin(&bn->climate[NP_EROSION], px, 0, pz);
    w = sampleDoublePerlin(&bn->climate[NP_WEIRDNESS], px, 0, pz);

    t = sampleDoublePerlin(&bn->climate[NP_TEMPERATURE], px, 0, pz);
    h = sampleDoublePerlin(&bn->climate[NP_HUMIDITY], px, 0, pz);

    int64_t l_np[6];
    int64_t *p_np = np ? np : l_np;
    p_np[0] = (int64_t)(10000.0F*t);
    p_np[1] = (int64_t)(10000.0F*h);
    p_np[2] = (int64_t)(10000.0F*c);
    p_np[3] = (int64_t)(10000.0F*e);
    p_np[4] = (int64_t)(10000.0F*d);
    p_np[5] = (int64_t)(10000.0F*w);

    int id = none;
    if (!(sample_flags & SAMPLE_NO_BIOME))
        id = climateToBiome(bn->mc, (const uint64_t*)p_np, dat);
    return id;
}

__host__ __device__ void setupGenerator(Generator *g, int mc, uint32_t flags) {
    g->mc = mc;
    g->dim = DIM_UNDEF;
    g->flags = flags;
    g->seed = 0;
    g->sha = 0;

    initBiomeNoise(&g->bn, mc);
}

__host__ __device__ void xPerlinInit(PerlinNoise *noise, Xoroshiro *xr)
{
    int i = 0;
    //memset(noise, 0, sizeof(*noise));
    noise->a = xNextDouble(xr) * 256.0;
    noise->b = xNextDouble(xr) * 256.0;
    noise->c = xNextDouble(xr) * 256.0;
    noise->amplitude = 1.0;
    noise->lacunarity = 1.0;

    uint8_t *idx = noise->d;
    for (i = 0; i < 256; i++)
    {
        idx[i] = i;
    }
    for (i = 0; i < 256; i++)
    {
        int j = xNextInt(xr, 256 - i) + i;
        uint8_t n = idx[i];
        idx[i] = idx[j];
        idx[j] = n;
    }
    idx[256] = idx[0];
    double i2 = floor(noise->b);
    double d2 = noise->b - i2;
    noise->h2 = (int) i2;
    noise->d2 = d2;
    noise->t2 = d2*d2*d2 * (d2 * (d2*6.0-15.0) + 10.0);
}


__host__ __device__ int xOctaveInit(OctaveNoise *noise, Xoroshiro *xr, PerlinNoise *octaves,
        const double *amplitudes, int omin, int len, int nmax)
{
    static const uint64_t md5_octave_n[][2] = {
        {0xb198de63a8012672, 0x7b84cad43ef7b5a8}, // md5 "octave_-12"
        {0x0fd787bfbc403ec3, 0x74a4a31ca21b48b8}, // md5 "octave_-11"
        {0x36d326eed40efeb2, 0x5be9ce18223c636a}, // md5 "octave_-10"
        {0x082fe255f8be6631, 0x4e96119e22dedc81}, // md5 "octave_-9"
        {0x0ef68ec68504005e, 0x48b6bf93a2789640}, // md5 "octave_-8"
        {0xf11268128982754f, 0x257a1d670430b0aa}, // md5 "octave_-7"
        {0xe51c98ce7d1de664, 0x5f9478a733040c45}, // md5 "octave_-6"
        {0x6d7b49e7e429850a, 0x2e3063c622a24777}, // md5 "octave_-5"
        {0xbd90d5377ba1b762, 0xc07317d419a7548d}, // md5 "octave_-4"
        {0x53d39c6752dac858, 0xbcd1c5a80ab65b3e}, // md5 "octave_-3"
        {0xb4a24d7a84e7677b, 0x023ff9668e89b5c4}, // md5 "octave_-2"
        {0xdffa22b534c5f608, 0xb9b67517d3665ca9}, // md5 "octave_-1"
        {0xd50708086cef4d7c, 0x6e1651ecc7f43309}, // md5 "octave_0"
    };
    static const double lacuna_ini[] = { // -omin = 3..12
        1, .5, .25, 1./8, 1./16, 1./32, 1./64, 1./128, 1./256, 1./512, 1./1024,
        1./2048, 1./4096,
    };
    static const double persist_ini[] = { // len = 4..9
        0, 1, 2./3, 4./7, 8./15, 16./31, 32./63, 64./127, 128./255, 256./511,
    };
#if DEBUG
    if (-omin < 0 || -omin >= (int) (sizeof(lacuna_ini)/sizeof(double)) ||
        len < 0 || len >= (int) (sizeof(persist_ini)/sizeof(double)))
    {
        printf("Fatal: octave initialization out of range\n");
        //exit(1);
    }
#endif
    double lacuna = lacuna_ini[-omin];
    double persist = persist_ini[len];
    uint64_t xlo = xNextLong(xr);
    uint64_t xhi = xNextLong(xr);
    int i = 0, n = 0;

    for (; i < len && n != nmax; i++, lacuna *= 2.0, persist *= 0.5)
    {
        if (amplitudes[i] == 0)
            continue;
        Xoroshiro pxr;
        pxr.lo = xlo ^ md5_octave_n[12 + omin + i][0];
        pxr.hi = xhi ^ md5_octave_n[12 + omin + i][1];
        xPerlinInit(&octaves[n], &pxr);
        octaves[n].amplitude = amplitudes[i] * persist;
        octaves[n].lacunarity = lacuna;
        n++;
    }

    noise->octaves = octaves;
    noise->octcnt = n;
    return n;
}

__host__ __device__ int xDoublePerlinInit(DoublePerlinNoise *noise, Xoroshiro *xr,
        PerlinNoise *octaves, const double *amplitudes, int omin, int len, int nmax)
{
    int i, n = 0, na = -1, nb = -1;
    if (nmax > 0)
    {
        na = (nmax + 1) >> 1;
        nb = nmax - na;
    }
    n += xOctaveInit(&noise->octA, xr, octaves+n, amplitudes, omin, len, na);
    n += xOctaveInit(&noise->octB, xr, octaves+n, amplitudes, omin, len, nb);

    // trim amplitudes of zero
    for (i = len-1; i >= 0 && amplitudes[i] == 0.0; i--)
        len--;
    for (i = 0; amplitudes[i] == 0.0; i++)
        len--;
    static const double amp_ini[] = { // (5 ./ 3) * len / (len + 1), len = 2..9
        0, 5./6, 10./9, 15./12, 20./15, 25./18, 30./21, 35./24, 40./27, 45./30,
    };
    noise->amplitude = amp_ini[len];
    return n;
}

__host__ __device__ static int init_climate_seed(
    DoublePerlinNoise *dpn, PerlinNoise *oct,
    uint64_t xlo, uint64_t xhi, int large, int nptype, int nmax
    )
{
    Xoroshiro pxr;
    int n = 0;

    switch (nptype)
    {
    case NP_SHIFT: {
        static const double amp[] = {1, 1, 1, 0};
        // md5 "minecraft:offset"
        pxr.lo = xlo ^ 0x080518cf6af25384;
        pxr.hi = xhi ^ 0x3f3dfb40a54febd5;
        n += xDoublePerlinInit(dpn, &pxr, oct, amp, -3, 4, nmax);
        } break;

    case NP_TEMPERATURE: {
        static const double amp[] = {1.5, 0, 1, 0, 0, 0};
        // md5 "minecraft:temperature" or "minecraft:temperature_large"
        pxr.lo = xlo ^ (large ? 0x944b0073edf549db : 0x5c7e6b29735f0d7f);
        pxr.hi = xhi ^ (large ? 0x4ff44347e9d22b96 : 0xf7d86f1bbc734988);
        n += xDoublePerlinInit(dpn, &pxr, oct, amp, large ? -12 : -10, 6, nmax);
        } break;

    case NP_HUMIDITY: {
        static const double amp[] = {1, 1, 0, 0, 0, 0};
        // md5 "minecraft:vegetation" or "minecraft:vegetation_large"
        pxr.lo = xlo ^ (large ? 0x71b8ab943dbd5301 : 0x81bb4d22e8dc168e);
        pxr.hi = xhi ^ (large ? 0xbb63ddcf39ff7a2b : 0xf1c8b4bea16303cd);
        n += xDoublePerlinInit(dpn, &pxr, oct, amp, large ? -10 : -8, 6, nmax);
        } break;

    case NP_CONTINENTALNESS: {
        static const double amp[] = {1, 1, 2, 2, 2, 1, 1, 1, 1};
        // md5 "minecraft:continentalness" or "minecraft:continentalness_large"
        pxr.lo = xlo ^ (large ? 0x9a3f51a113fce8dc : 0x83886c9d0ae3a662);
        pxr.hi = xhi ^ (large ? 0xee2dbd157e5dcdad : 0xafa638a61b42e8ad);
        n += xDoublePerlinInit(dpn, &pxr, oct, amp, large ? -11 : -9, 9, nmax);
        } break;

    case NP_EROSION: {
        static const double amp[] = {1, 1, 0, 1, 1};
        // md5 "minecraft:erosion" or "minecraft:erosion_large"
        pxr.lo = xlo ^ (large ? 0x8c984b1f8702a951 : 0xd02491e6058f6fd8);
        pxr.hi = xhi ^ (large ? 0xead7b1f92bae535f : 0x4792512c94c17a80);
        n += xDoublePerlinInit(dpn, &pxr, oct, amp, large ? -11 : -9, 5, nmax);
        } break;

    case NP_WEIRDNESS: {
        static const double amp[] = {1, 2, 1, 0, 0, 0};
        // md5 "minecraft:ridge"
        pxr.lo = xlo ^ 0xefc8ef4d36102b34;
        pxr.hi = xhi ^ 0x1beeeb324a0f24ea;
        n += xDoublePerlinInit(dpn, &pxr, oct, amp, -7, 6, nmax);
        } break;

    default:
        printf("unsupported climate parameter %d\n", nptype);
    }
    return n;
}

__host__ __device__ void setBiomeSeed(BiomeNoise *bn, uint64_t seed, int large)
{
    Xoroshiro pxr;
    xSetSeed(&pxr, seed);
    uint64_t xlo = xNextLong(&pxr);
    uint64_t xhi = xNextLong(&pxr);

    int n = 0, i = 0;
    for (; i < NP_MAX; i++)
        n += init_climate_seed(&bn->climate[i], bn->oct+n, xlo, xhi, large, i, -1);

    if ((size_t)n > sizeof(bn->oct) / sizeof(*bn->oct))
    {
        printf("setBiomeSeed(): BiomeNoise is malformed, buffer too small\n");
    }
    bn->nptype = -1;
}

__host__ __device__ void applySeed(Generator *g, int dim, uint64_t seed)
{
    g->dim = dim;
    g->seed = seed;
    g->sha = 0;

    if (dim == DIM_OVERWORLD)
    {
        setBiomeSeed(&g->bn, seed, g->flags & LARGE_BIOMES);
    }
}

STRUCT(StructureConfig)
{
    int32_t salt;
    int8_t  regionSize;
    int8_t  chunkRange;
    uint8_t structType;
    uint8_t properties;
    float   rarity;
};

__host__ __device__ static inline ATTR(const)
Pos getFeatureChunkInRegion(StructureConfig config, uint64_t seed, int regX, int regZ)
{
    /*
    // Vanilla like implementation.
    setSeed(&seed, regX*341873128712 + regZ*132897987541 + seed + config.salt);

    Pos pos;
    pos.x = nextInt(&seed, 24);
    pos.z = nextInt(&seed, 24);
    */
    Pos pos;
    const uint64_t K = 0x5deece66dULL;
    const uint64_t M = (1ULL << 48) - 1;
    const uint64_t b = 0xb;

    // set seed
    seed = seed + regX*341873128712ULL + regZ*132897987541ULL + config.salt;
    seed = (seed ^ K);
    seed = (seed * K + b) & M;

    uint64_t r = config.chunkRange;
    if (r & (r-1))
    {
        pos.x = (int)(seed >> 17) % r;
        seed = (seed * K + b) & M;
        pos.z = (int)(seed >> 17) % r;
    }
    else
    {
        // Java RNG treats powers of 2 as a special case.
        pos.x = (int)((r * (seed >> 17)) >> 31);
        seed = (seed * K + b) & M;
        pos.z = (int)((r * (seed >> 17)) >> 31);
    }

    return pos;
}

__host__ __device__ static inline ATTR(const)
Pos getFeaturePos(StructureConfig config, uint64_t seed, int regX, int regZ)
{
    Pos pos = getFeatureChunkInRegion(config, seed, regX, regZ);

    pos.x = (int)(((uint64_t)regX*config.regionSize + pos.x) << 4);
    pos.z = (int)(((uint64_t)regZ*config.regionSize + pos.z) << 4);
    return pos;
}

__host__ __device__ int getStructureConfig(int structureType, int mc, StructureConfig *sconf)
{
    static const StructureConfig
    // for desert pyramids, jungle temples, witch huts and igloos prior to 1.13
    s_feature               = { 14357617, 32, 24, Feature,          0,0},
    s_igloo_112             = { 14357617, 32, 24, Igloo,            0,0},
    s_swamp_hut_112         = { 14357617, 32, 24, Swamp_Hut,        0,0},
    s_desert_pyramid_112    = { 14357617, 32, 24, Desert_Pyramid,   0,0},
    s_jungle_temple_112     = { 14357617, 32, 24, Jungle_Pyramid,   0,0},
    // ocean features before 1.16
    s_ocean_ruin_115        = { 14357621, 16,  8, Ocean_Ruin,       0,0},
    s_shipwreck_115         = {165745295, 16,  8, Shipwreck,        0,0},
    // 1.13 separated feature seeds by type
    s_desert_pyramid        = { 14357617, 32, 24, Desert_Pyramid,   0,0},
    s_igloo                 = { 14357618, 32, 24, Igloo,            0,0},
    s_jungle_temple         = { 14357619, 32, 24, Jungle_Pyramid,   0,0},
    s_swamp_hut             = { 14357620, 32, 24, Swamp_Hut,        0,0},
    s_outpost               = {165745296, 32, 24, Outpost,          0,0},
    s_village_117           = { 10387312, 32, 24, Village,          0,0},
    s_village               = { 10387312, 34, 26, Village,          0,0},
    s_ocean_ruin            = { 14357621, 20, 12, Ocean_Ruin,       0,0},
    s_shipwreck             = {165745295, 24, 20, Shipwreck,        0,0},
    s_monument              = { 10387313, 32, 27, Monument,         STRUCT_TRIANGULAR,0},
    s_mansion               = { 10387319, 80, 60, Mansion,          STRUCT_TRIANGULAR,0},
    s_ruined_portal         = { 34222645, 40, 25, Ruined_Portal,    0,0},
    s_ruined_portal_n       = { 34222645, 40, 25, Ruined_Portal,    STRUCT_NETHER,0},
    s_ruined_portal_n_117   = { 34222645, 25, 15, Ruined_Portal_N,  STRUCT_NETHER,0},
    s_ancient_city          = { 20083232, 24, 16, Ancient_City,     0,0},
    s_trail_ruins           = { 83469867, 34, 26, Trail_Ruins,      0,0},
    s_trial_chambers        = { 94251327, 34, 22, Trial_Chambers,   0,0},
    s_treasure              = { 10387320,  1,  1, Treasure,         STRUCT_CHUNK,0},
    s_mineshaft             = {        0,  1,  1, Mineshaft,        STRUCT_CHUNK,0},
    s_desert_well_115       = {    30010,  1,  1, Desert_Well,      STRUCT_CHUNK, 1.f/1000},
    s_desert_well_117       = {    40013,  1,  1, Desert_Well,      STRUCT_CHUNK, 1.f/1000},
    s_desert_well           = {    40002,  1,  1, Desert_Well,      STRUCT_CHUNK, 1.f/1000},
    s_geode_117             = {    20000,  1,  1, Geode,            STRUCT_CHUNK, 1.f/24},
    s_geode                 = {    20002,  1,  1, Geode,            STRUCT_CHUNK, 1.f/24},
    // nether and end structures
    s_fortress_115          = {        0, 16,  8, Fortress,         STRUCT_NETHER,0},
    s_fortress              = { 30084232, 27, 23, Fortress,         STRUCT_NETHER,0},
    s_bastion               = { 30084232, 27, 23, Bastion,          STRUCT_NETHER,0},
    s_end_city              = { 10387313, 20,  9, End_City,         STRUCT_END|STRUCT_TRIANGULAR,0},
    // for the scattered return gateways
    s_end_gateway_115       = {    30000,  1,  1, End_Gateway,      STRUCT_END|STRUCT_CHUNK, 700},
    s_end_gateway_116       = {    40013,  1,  1, End_Gateway,      STRUCT_END|STRUCT_CHUNK, 700},
    s_end_gateway_117       = {    40013,  1,  1, End_Gateway,      STRUCT_END|STRUCT_CHUNK, 1.f/700},
    s_end_gateway           = {    40000,  1,  1, End_Gateway,      STRUCT_END|STRUCT_CHUNK, 1.f/700},
    s_end_island_115        = {        0,  1,  1, End_Island,       STRUCT_END|STRUCT_CHUNK, 14},
    s_end_island            = {        0,  1,  1, End_Island,       STRUCT_END|STRUCT_CHUNK, 1.f/14}
    ;

    switch (structureType)
    {
    case Feature:
        *sconf = s_feature;
        return mc <= MC_1_12;
    case Desert_Pyramid:
        *sconf = mc <= MC_1_12 ? s_desert_pyramid_112 : s_desert_pyramid;
        return mc >= MC_1_3;
    case Jungle_Pyramid:
        *sconf = mc <= MC_1_12 ? s_jungle_temple_112 : s_jungle_temple;
        return mc >= MC_1_3;
    case Swamp_Hut:
        *sconf = mc <= MC_1_12 ? s_swamp_hut_112 : s_swamp_hut;
        return mc >= MC_1_4;
    case Igloo:
        *sconf = mc <= MC_1_12 ? s_igloo_112 : s_igloo;
        return mc >= MC_1_9;
    case Village:
        *sconf = mc <= MC_1_17 ? s_village_117 : s_village;
        return mc >= MC_B1_8;
    case Ocean_Ruin:
        *sconf = mc <= MC_1_15 ? s_ocean_ruin_115 : s_ocean_ruin;
        return mc >= MC_1_13;
    case Shipwreck:
        *sconf = mc <= MC_1_15 ? s_shipwreck_115 : s_shipwreck;
        return mc >= MC_1_13;
    case Ruined_Portal:
        *sconf = s_ruined_portal;
        return mc >= MC_1_16_1;
    case Ruined_Portal_N:
        *sconf = mc <= MC_1_17 ? s_ruined_portal_n_117 : s_ruined_portal_n;
        return mc >= MC_1_16_1;
    case Monument:
        *sconf = s_monument;
        return mc >= MC_1_8;
    case End_City:
        *sconf = s_end_city;
        return mc >= MC_1_9;
    case Mansion:
        *sconf = s_mansion;
        return mc >= MC_1_11;
    case Outpost:
        *sconf = s_outpost;
        return mc >= MC_1_14;
    case Ancient_City:
        *sconf = s_ancient_city;
        return mc >= MC_1_19_2;
    case Treasure:
        *sconf = s_treasure;
        return mc >= MC_1_13;
    case Mineshaft:
        *sconf = s_mineshaft;
        return mc >= MC_B1_8;
    case Fortress:
        *sconf = mc <= MC_1_15 ? s_fortress_115 : s_fortress;
        return mc >= MC_1_0;
    case Bastion:
        *sconf = s_bastion;
        return mc >= MC_1_16_1;
    case End_Gateway:
        if      (mc <= MC_1_15) *sconf = s_end_gateway_115;
        else if (mc <= MC_1_16) *sconf = s_end_gateway_116;
        else if (mc <= MC_1_17) *sconf = s_end_gateway_117;
        else                    *sconf = s_end_gateway;
        // 1.11 and 1.12 generate gateways using a random source that passed
        // the block filling, making them much more difficult to predict
        return mc >= MC_1_13;
    case End_Island:
        if      (mc <= MC_1_15) *sconf = s_end_island_115;
        else                    *sconf = s_end_island;
        return mc >= MC_1_13; // we only support decorator features for 1.13+
    case Desert_Well:
        if      (mc <= MC_1_15) *sconf = s_desert_well_115;
        else if (mc <= MC_1_17) *sconf = s_desert_well_117;
        else                    *sconf = s_desert_well;
        // wells were introduced in 1.2, but we only support decorator features
        // for 1.13+
        return mc >= MC_1_13;
    case Geode:
        *sconf = mc <= MC_1_17 ? s_geode_117 : s_geode;
        return mc >= MC_1_17;
    case Trail_Ruins:
        *sconf = s_trail_ruins;
        return mc >= MC_1_20;
    case Trial_Chambers:
        *sconf = s_trial_chambers;
        return mc >= MC_1_21;
    default:
        memset(sconf, 0, sizeof(StructureConfig));
        return 0;
    }
}

STRUCT(StructureVariant)
{
    uint8_t abandoned   :1; // is zombie village
    uint8_t giant       :1; // giant portal variant
    uint8_t underground :1; // underground portal
    uint8_t airpocket   :1; // portal with air pocket
    uint8_t basement    :1; // igloo with basement
    uint8_t cracked     :1; // geode with crack
    uint8_t size;           // geode size | igloo middel pieces
    uint8_t start;          // starting piece index
    short   biome;          // biome variant
    uint8_t rotation;       // 0:0, 1:cw90, 2:cw180, 3:cw270=ccw90
    uint8_t mirror;
    int16_t x, y, z;
    int16_t sx, sy, sz;
};

__host__ __device__ const Layer *getLayerForScale(const Generator *g, int scale)
{
    if (g->mc > MC_1_17)
        return NULL;
    switch (scale)
    {
    case 0:   return g->entry;
    case 1:   return g->ls.entry_1;
    case 4:   return g->ls.entry_4;
    case 16:  return g->ls.entry_16;
    case 64:  return g->ls.entry_64;
    case 256: return g->ls.entry_256;
    default:
        return NULL;
    }
}

__host__ __device__ static void getMaxArea(
    const Layer *layer, int areaX, int areaZ, int *maxX, int *maxZ, size_t *siz)
{
    if (layer == NULL)
        return;

    areaX += layer->edge;
    areaZ += layer->edge;

    // multi-layers and zoom-layers use a temporary copy of their parent area
    if (layer->p2 || layer->zoom != 1)
        *siz += areaX * areaZ;

    if (areaX > *maxX) *maxX = areaX;
    if (areaZ > *maxZ) *maxZ = areaZ;

    if (layer->zoom == 2)
    {
        areaX >>= 1;
        areaZ >>= 1;
    }
    else if (layer->zoom == 4)
    {
        areaX >>= 2;
        areaZ >>= 2;
    }

    getMaxArea(layer->p, areaX, areaZ, maxX, maxZ, siz);
    if (layer->p2)
        getMaxArea(layer->p2, areaX, areaZ, maxX, maxZ, siz);
}

__host__ __device__ static inline void getVoronoiCell(uint64_t sha, int a, int b, int c,
        int *x, int *y, int *z)
{
    uint64_t s = sha;
    s = mcStepSeed(s, a);
    s = mcStepSeed(s, b);
    s = mcStepSeed(s, c);
    s = mcStepSeed(s, a);
    s = mcStepSeed(s, b);
    s = mcStepSeed(s, c);

    *x = (((s >> 24) & 1023) - 512) * 36;
    s = mcStepSeed(s, sha);
    *y = (((s >> 24) & 1023) - 512) * 36;
    s = mcStepSeed(s, sha);
    *z = (((s >> 24) & 1023) - 512) * 36;
}

__host__ __device__ void voronoiAccess3D(uint64_t sha, int x, int y, int z, int *x4, int *y4, int *z4)
{
    x -= 2;
    y -= 2;
    z -= 2;
    int pX = x >> 2;
    int pY = y >> 2;
    int pZ = z >> 2;
    int dx = (x & 3) * 10240;
    int dy = (y & 3) * 10240;
    int dz = (z & 3) * 10240;
    int ax = 0, ay = 0, az = 0;
    uint64_t dmin = (uint64_t)-1;
    int i;

    for (i = 0; i < 8; i++)
    {
        int bx = (i & 4) != 0;
        int by = (i & 2) != 0;
        int bz = (i & 1) != 0;
        int cx = pX + bx;
        int cy = pY + by;
        int cz = pZ + bz;
        int rx, ry, rz;

        getVoronoiCell(sha, cx, cy, cz, &rx, &ry, &rz);

        rx += dx - 40*1024*bx;
        ry += dy - 40*1024*by;
        rz += dz - 40*1024*bz;

        uint64_t d = rx*(uint64_t)rx + ry*(uint64_t)ry + rz*(uint64_t)rz;
        if (d < dmin)
        {
            dmin = d;
            ax = cx;
            ay = cy;
            az = cz;
        }
    }

    if (x4) *x4 = ax;
    if (y4) *y4 = ay;
    if (z4) *z4 = az;
}

__host__ __device__ static void genBiomeNoise3D(const BiomeNoise *bn, int *out, Range r, int opt)
{
    uint64_t dat = 0;
    uint64_t *p_dat = opt ? &dat : NULL;
    uint32_t flags = opt ? SAMPLE_NO_SHIFT : 0;
    int i, j, k;
    int *p = out;
    int scale = r.scale > 4 ? r.scale / 4 : 1;
    int mid = scale / 2;
    for (k = 0; k < r.sy; k++)
    {
        int yk = (r.y+k);
        for (j = 0; j < r.sz; j++)
        {
            int zj = (r.z+j)*scale + mid;
            for (i = 0; i < r.sx; i++)
            {
                int xi = (r.x+i)*scale + mid;
                *p = sampleBiomeNoise(bn, NULL, xi, yk, zj, p_dat, flags);
                p++;
            }
        }
    }
}

__host__ __device__ Range getVoronoiSrcRange(Range r)
{
    if (r.scale != 1)
    {
        printf("getVoronoiSrcRange() expects input range with scale 1:1\n");
    }

    Range s; // output has scale 1:4
    int x = r.x - 2;
    int z = r.z - 2;
    s.scale = 4;
    s.x = x >> 2;
    s.z = z >> 2;
    s.sx = ((x + r.sx) >> 2) - s.x + 2;
    s.sz = ((z + r.sz) >> 2) - s.z + 2;
    if (r.sy < 1)
    {
        s.y = s.sy = 0;
    }
    else
    {
        int ty = r.y - 2;
        s.y = ty >> 2;
        s.sy = ((ty + r.sy) >> 2) - s.y + 2;
    }
    return s;
}

__host__ __device__ int genBiomeNoiseScaled(const BiomeNoise *bn, int *out, Range r, uint64_t sha)
{
    if (r.sy == 0)
        r.sy = 1;

    uint64_t siz = (uint64_t)r.sx*r.sy*r.sz;
    int i, j, k;

    if (r.scale == 1)
    {
        Range s = getVoronoiSrcRange(r);
        int *src;
        if (siz > 1)
        {   // the source range is large enough that we can try optimizing
            src = out + siz;
            genBiomeNoise3D(bn, src, s, 0);
        }
        else
        {
            src = NULL;
        }

        int *p = out;
        for (k = 0; k < r.sy; k++)
        {
            for (j = 0; j < r.sz; j++)
            {
                for (i = 0; i < r.sx; i++)
                {
                    int x4, z4, y4;
                    voronoiAccess3D(sha, r.x+i, r.y+k, r.z+j, &x4, &y4, &z4);
                    if (src)
                    {
                        x4 -= s.x; y4 -= s.y; z4 -= s.z;
                        *p = src[(int64_t)y4*s.sx*s.sz + (int64_t)z4*s.sx + x4];
                    }
                    else
                    {
                        *p = sampleBiomeNoise(bn, 0, x4, y4, z4, 0, 0);
                    }
                    p++;
                }
            }
        }
    }
    else
    {
        // There is (was?) an optimization that causes MC-241546, and should
        // not be enabled for accurate results. However, if the scale is higher
        // than 1:4, the accuracy becomes questionable anyway. Furthermore
        // situations that want to use a higher scale are usually better off
        // with a faster, if imperfect, result.
        genBiomeNoise3D(bn, out, r, r.scale > 4);
    }
    return 0;
}

__host__ __device__ int genBiomes(const Generator *g, int *cache, Range r)
{
    int err = 1;
    int64_t i, k;

    return genBiomeNoiseScaled(&g->bn, cache, r, g->sha);
    return err;
}

__host__ __device__ size_t getMinLayerCacheSize(const Layer *layer, int sizeX, int sizeZ)
{
    int maxX = sizeX, maxZ = sizeZ;
    size_t bufsiz = 0;
    getMaxArea(layer, sizeX, sizeZ, &maxX, &maxZ, &bufsiz);
    return bufsiz + maxX * (size_t)maxZ;
}

__host__ __device__ inline static
uint64_t chunkGenerateRnd(uint64_t worldSeed, int chunkX, int chunkZ)
{
    uint64_t rnd;
    setSeed(&rnd, worldSeed);
    rnd = (nextLong(&rnd) * chunkX) ^ (nextLong(&rnd) * chunkZ) ^ worldSeed;
    setSeed(&rnd, rnd);
    return rnd;
}

__host__ __device__ size_t getMinCacheSize(const Generator *g, int scale, int sx, int sy, int sz)
{
    if (sy == 0)
        sy = 1;
    size_t len = (size_t)sx * sz * sy;
    if (g->mc <= MC_B1_7 && scale <= 4 && !(g->flags & NO_BETA_OCEAN))
    {
        int cellwidth = scale >> 1;
        int smin = (sx < sz ? sx : sz);
        int slen = ((smin >> (2 >> cellwidth)) + 1) * 2 + 1;
        len += slen * sizeof(SeaLevelColumnNoiseBeta);
    }
    else if (g->mc >= MC_B1_8 && g->mc <= MC_1_17 && g->dim == DIM_OVERWORLD)
    {   // recursively check the layer stack for the max buffer
        const Layer *entry = getLayerForScale(g, scale);
        if (!entry) {
            printf("getMinCacheSize(): failed to determine scaled entry\n");
            //exit(1);
        }
        size_t len2d = getMinLayerCacheSize(entry, sx, sz);
        len += len2d - sx*sz;
    }
    else if ((g->mc >= MC_1_18 || g->dim != DIM_OVERWORLD) && scale <= 1)
    {   // allocate space for temporary copy of voronoi source
        sx = ((sx+3) >> 2) + 2;
        sy = ((sy+3) >> 2) + 2;
        sz = ((sz+3) >> 2) + 2;
        len += sx * sy * sz;
    }

    return len;
}

#include <cuda.h>

__host__ __device__ int *allocCache(const Generator *g, Range r)
{
  cudaError_t err = cudaSuccess;
	size_t len = getMinCacheSize(g, r.scale, r.sx, r.sy, r.sz);
  int *__temp = (int *)malloc(len * sizeof(int));
	for (size_t __idx = 0; __idx < len; ++__idx) __temp[__idx] = 0;
	return __temp;
}

__host__ __device__ int getBiomeAt(const Generator *g, int scale, int x, int y, int z)
{
    Range r = {scale, x, z, 1, 1, y, 1};
    //int *ids = allocCache(g, r);
    int ids[128];
    int id = genBiomes(g, ids, r);
    if (id == 0)
        id = ids[0];
    else
        id = none;
    // free(ids);
    return id;
}

__host__ __device__ int isViableFeatureBiome(int mc, int structureType, int biomeID) {
    return biomeID == plains || biomeID == desert || biomeID == savanna || biomeID == taiga || biomeID == snowy_taiga || biomeID == meadow;
}

__host__ __device__ int getVariant(StructureVariant *r, int structType, int mc, uint64_t seed,
        int x, int z, int biomeID)
{
    int t;
    char sx, sy, sz;
    uint64_t rng = chunkGenerateRnd(seed, x >> 4, z >> 4);

    memset(r, 0, sizeof(*r));
    r->start = -1;
    r->biome = -1;
    r->y = 320;

    if (!isViableFeatureBiome(mc, Village, biomeID))
        return 0;

    r->biome = biomeID;
    r->rotation = nextInt(&rng, 4);
    switch (biomeID)
    {
    case meadow:
        r->biome = plains;
        // fallthrough
    case plains:
        t = nextInt(&rng, 204);
        if      (t <  50) { r->start = 0; sx =  9; sy = 4; sz =  9; } // plains_fountain_01
        else if (t < 100) { r->start = 1; sx = 10; sy = 7; sz = 10; } // plains_meeting_point_1
        else if (t < 150) { r->start = 2; sx =  8; sy = 5; sz = 15; } // plains_meeting_point_2
        else if (t < 200) { r->start = 3; sx = 11; sy = 9; sz = 11; } // plains_meeting_point_3
        else if (t < 201) { r->start = 0; sx =  9; sy = 4; sz =  9; r->abandoned = 1; }
        else if (t < 202) { r->start = 1; sx = 10; sy = 7; sz = 10; r->abandoned = 1; }
        else if (t < 203) { r->start = 2; sx =  8; sy = 5; sz = 15; r->abandoned = 1; }
        else if (t < 204) { r->start = 3; sx = 11; sy = 9; sz = 11; r->abandoned = 1; }
        else UNREACHABLE();
        break;
    case desert:
        t = nextInt(&rng, 250);
        if      (t <  98) { r->start = 1; sx = 17; sy = 6; sz =  9; } // desert_meeting_point_1
        else if (t < 196) { r->start = 2; sx = 12; sy = 6; sz = 12; } // desert_meeting_point_2
        else if (t < 245) { r->start = 3; sx = 15; sy = 6; sz = 15; } // desert_meeting_point_3
        else if (t < 247) { r->start = 1; sx = 17; sy = 6; sz =  9; r->abandoned = 1; }
        else if (t < 249) { r->start = 2; sx = 12; sy = 6; sz = 12; r->abandoned = 1; }
        else if (t < 250) { r->start = 3; sx = 15; sy = 6; sz = 15; r->abandoned = 1; }
        else UNREACHABLE();
        break;
    case savanna:
        t = nextInt(&rng, 459);
        if      (t < 100) { r->start = 1; sx = 14; sy = 5; sz = 12; } // savanna_meeting_point_1
        else if (t < 150) { r->start = 2; sx = 11; sy = 6; sz = 11; } // savanna_meeting_point_2
        else if (t < 300) { r->start = 3; sx =  9; sy = 6; sz = 11; } // savanna_meeting_point_3
        else if (t < 450) { r->start = 4; sx =  9; sy = 6; sz =  9; } // savanna_meeting_point_4
        else if (t < 452) { r->start = 1; sx = 14; sy = 5; sz = 12; r->abandoned = 1; }
        else if (t < 453) { r->start = 2; sx = 11; sy = 6; sz = 11; r->abandoned = 1; }
        else if (t < 456) { r->start = 3; sx =  9; sy = 6; sz = 11; r->abandoned = 1; }
        else if (t < 459) { r->start = 4; sx =  9; sy = 6; sz =  9; r->abandoned = 1; }
        else UNREACHABLE();
        break;
    case taiga:
        t = nextInt(&rng, 100);
        if      (t <  49) { r->start = 1; sx = 22; sy = 3; sz = 18; } // taiga_meeting_point_1
        else if (t <  98) { r->start = 2; sx =  9; sy = 7; sz =  9; } // taiga_meeting_point_2
        else if (t <  99) { r->start = 1; sx = 22; sy = 3; sz = 18; r->abandoned = 1; }
        else if (t < 100) { r->start = 2; sx =  9; sy = 7; sz =  9; r->abandoned = 1; }
        else UNREACHABLE();
        break;
    case snowy_tundra:
        t = nextInt(&rng, 306);
        if      (t < 100) { r->start = 1; sx = 12; sy = 8; sz =  8; } // snowy_meeting_point_1
        else if (t < 150) { r->start = 2; sx = 11; sy = 5; sz =  9; } // snowy_meeting_point_2
        else if (t < 300) { r->start = 3; sx =  7; sy = 7; sz =  7; } // snowy_meeting_point_3
        else if (t < 302) { r->start = 1; sx = 12; sy = 8; sz =  8; r->abandoned = 1; }
        else if (t < 303) { r->start = 2; sx = 11; sy = 5; sz =  9; r->abandoned = 1; }
        else if (t < 306) { r->start = 3; sx =  7; sy = 7; sz =  7; r->abandoned = 1; }
        else UNREACHABLE();
        break;
    default:
        sx = sy = sz = 0;
        return 0;
    }

    int lookups[4][4] = {
        {0, 0, sx, sz},
        {1-sz, 0, sz, sx},
        {1-sx, 1-sz, sx, sz},
        {0, 1-sx, sz, sx},
    };
    r->sy = sy;

    r->x = lookups[r->rotation][0];
    r->z = lookups[r->rotation][1];
    r->sx = lookups[r->rotation][2];
    r->sz = lookups[r->rotation][3];

    return 1;
}

__host__ __device__ int isViableStructurePos(int structureType, Generator *g, int x, int z, uint32_t flags)
{
    int approx = 0; // enables approximation levels
    int viable = 0;

    int64_t chunkX = x >> 4;
    int64_t chunkZ = z >> 4;

    // Structures are positioned at the chunk origin, but the biome check is
    // performed near the middle of the chunk [(9,9) in 1.13, TODO: check 1.7]
    // In 1.16 the biome check is always performed at (2,2) with layer scale=4.
    int sampleX, sampleZ, sampleY;
    int id;

    // Overworld

    const int vv[] = { plains, desert, savanna, taiga, snowy_tundra };
        // In 1.18 village types are checked separtely...

    size_t i;
    for (i = 0; i < sizeof(vv)/sizeof(int); i++) {
        StructureVariant sv;
        getVariant(&sv, Village, g->mc, g->seed, x, z, vv[i]);
        sampleX = (chunkX*32 + 2*sv.x + sv.sx-1) / 2 >> 2;
        sampleZ = (chunkZ*32 + 2*sv.z + sv.sz-1) / 2 >> 2;
        sampleY = 319 >> 2;
        id = getBiomeAt(g, 0, sampleX, sampleY, sampleZ);
        if (id == vv[i] || (id == meadow && vv[i] == plains)) {
            return 1;
        }
    }
    return 0;
}

__device__ int best = 9999;
__device__ __managed__ unsigned long long int checked = 0;

__global__ void kernel(uint64_t s, uint64_t *out, uint64_t *out_villages) {
    uint64_t input_seed = blockDim.x * blockIdx.x + threadIdx.x + s;
    //atomicAdd(&checked, 1ull);

    int structType = Village;
    int mc = MC_1_20;

    Generator g;
    setupGenerator(&g, mc, 0);

    StructureConfig sconf;
    getStructureConfig(Village, mc, &sconf);

    int region_size = 34;

    int size = 5 * region_size;

    uint64_t seed = input_seed;
    //printf("%" PRIu64 "\n", seed);
    applySeed(&g, DIM_OVERWORLD, seed);
    int villages = 0;
    int i = 0;
    bool found = false;
    for (int rx = -6; rx < 6; rx++) {
        for (int rz = -6; rz < 6; rz++) {
            Pos p = getFeaturePos(sconf, seed, rx, rz);
            found = isViableStructurePos(structType, &g, p.x, p.z, 0);

           	if (found) {
           		//villages++;
       			return;
       		}
        }
    }
    out[blockDim.x * blockIdx.x + threadIdx.x + s] = seed;
    out_villages[blockDim.x * blockIdx.x + threadIdx.x + s] = villages;
    //printf("%d\n", villages);
    //printf("Found new best: %" PRIi64 " %d\n", seed, villages);
}

#include <time.h>

#ifdef __GNUC__

#include <unistd.h>
#include <sys/time.h>

#endif

struct checkpoint_vars {
    unsigned long long offset;
    time_t elapsed_chkpoint;
};

time_t elapsed_chkpoint = 0;

int main(int argc, char **argv) {
    int block_min = atoi(argv[1]);
    int block_max = atoi(argv[2]);
    int device = atoi(argv[3]);
    uint64_t offsetStart = 0;
    uint64_t *out;
    uint64_t *out_villages;
    //GPU Params
	int blocks = 32768;
	int threads = 32;
    //BOINC
  	#ifdef BOINC
        BOINC_OPTIONS options;
        boinc_options_defaults(options);
	    options.normal_thread_priority = true;
        boinc_init_options(&options);
        FILE *checkpoint_data = boinc_fopen("checkpoint.txt", "rb");
        if(!checkpoint_data){
            fprintf(stderr, "No checkpoint to load\n");

        }
        else{
            boinc_begin_critical_section();
            struct checkpoint_vars data_store;
            fread(&data_store, sizeof(data_store), 1, checkpoint_data);
            offsetStart = data_store.offset;
            elapsed_chkpoint = data_store.elapsed_chkpoint;
            fprintf(stderr, "Checkpoint loaded, task time %d s, seed pos: %llu\n", elapsed_chkpoint, offsetStart);
            fclose(checkpoint_data);
            boinc_end_critical_section();
        }
    #endif
    cudaSetDevice(device);
    cudaMallocManaged(&out, (blocks * threads) * sizeof(*out));
    cudaMallocManaged(&out_villages, (blocks * threads) * sizeof(*out_villages));


    time_t start = time(NULL);

    //gettimeofday(&start, NULL);

	printf("starting...\n");
    uint64_t checkpointTemp = 0;
    FILE* seedsout = fopen("seeds.txt", "w+");
    for (uint64_t s = (uint64_t)block_min + offsetStart; s < (uint64_t)block_max; s++) {

        kernel<<<blocks, threads>>>(blocks * threads * s, out, out_villages);
        cudaDeviceSynchronize();
        checkpointTemp += 1;
        #ifdef BOINC
        if(checkpointTemp >= 15 || boinc_time_to_checkpoint()){
            time_t elapsed = time(NULL) - start;
            boinc_begin_critical_section(); // Boinc should not interrupt this
            
            // Checkpointing section below
            boinc_delete_file("checkpoint.txt"); // Don't touch, same func as normal fdel
            FILE *checkpoint_data = boinc_fopen("checkpoint.txt", "wb");
            struct checkpoint_vars data_store;
            data_store.offset = s;
            data_store.elapsed_chkpoint = elapsed_chkpoint + elapsed;
            fwrite(&data_store, sizeof(data_store), 1, checkpoint_data);
            fclose(checkpoint_data);
            checkpointTemp = 0;
            boinc_end_critical_section();
            boinc_checkpoint_completed(); // Checkpointing completed
        }
        #endif
        for (unsigned long long i = 0; i < blocks * threads; i++){
            if(out[i] > 0)
			    fprintf(seedsout,"s: %llu, v: %llu\n", out[i], out_villages[i]);
		}
		fflush(seedsout);
    }


    time_t end = time(NULL);

    double time_taken = end - start ; // in seconds
    checked = blocks*threads*(block_max - block_min);
    fprintf(stderr, "checked = %" PRIu64 "\n", checked);
    fprintf(stderr, "time taken = %f\n", time_taken);

	double seeds_per_second = checked / time_taken;
	double speedup = seeds_per_second / 199000;
	fprintf(stderr, "seeds per second: %f\n", seeds_per_second);
	fprintf(stderr, "speedup: %fx\n", speedup);
}
