cmake_minimum_required(VERSION 3.0.2)
project(panoptic_ros)


set(CATKIN_PACKAGE_DEPENDENCIES
  cv_bridge
  rospy
  panoptic_models
  mask2former
)
## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
    ${CATKIN_PACKAGE_DEPENDENCIES}
)


catkin_python_setup()

catkin_package(
  INCLUDE_DIRS
  LIBRARIES
  CATKIN_DEPENDS
    ${CATKIN_PACKAGE_DEPENDENCIES}
  DEPENDS
)


###########
## Build ##
###########

#############
## Install ##
#############
install(
  TARGETS
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

catkin_install_python(PROGRAMS
  panoptic_ros/estimator_node.py
  test/test_stream_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)