file(GLOB_RECURSE srcs CONFIGURE_DEPENDS src/*.cpp include/*.h)
add_library(sfr STATIC ${srcs})
target_include_directories(sfr PUBLIC include)