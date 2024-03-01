cmake_minimum_required(VERSION 3.0.2)
project(panoptic_models)


set(CATKIN_PACKAGE_DEPENDENCIES
  cv_bridge
  rospy
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
