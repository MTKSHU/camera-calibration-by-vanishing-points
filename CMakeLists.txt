# set project's name
PROJECT( VanishingPoint )

###############################################################################
# CMake settings
CMAKE_MINIMUM_REQUIRED(VERSION 2.8.3)

# OpenCV
#set (OpenCV_DIR /usr/local/share/OpenCV)
FIND_PACKAGE(OpenCV 2.4.13.6 REQUIRED)

message(STATUS "OpenCV library status:")
message(STATUS " config: ${OPENCV_DIR}")
message(STATUS " version: ${OpenV_VERSION}")
message(STATUS " libraries: ${OpenCV_LIBS}")
message(STATUS " include_path: ${OpenCV_INCLUDE_DIRS}")

FILE(GLOB_RECURSE HDRS_FILES "src/*.h" "src/*.hpp")
FILE(GLOB_RECURSE SRCS_FILES "src/*.c" "src/*.cpp")

include_directories (${OpenCV_INCLUDE_DIRS})
ADD_EXECUTABLE(${PROJECT_NAME} ${SRCS_FILES} ${HDRS_FILES})
TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${OpenCV_LIBS})
