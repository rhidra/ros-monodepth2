cmake_minimum_required(VERSION 3.0.2)
project(ros-monodepth2)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)

# add_executable(octo_node src/octo_node.cpp)
# target_link_libraries(octo_node ${catkin_LIBRARIES})
# add_dependencies(octo_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

catkin_install_python(PROGRAMS run.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
