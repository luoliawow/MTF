file(GLOB_RECURSE srcs CONFIGURE_DEPENDS src/*.cpp include/*.h)
add_executable(sfrmain ${srcs})
target_include_directories(sfrmain PUBLIC include)

target_link_libraries(sfrmain PUBLIC sfr)