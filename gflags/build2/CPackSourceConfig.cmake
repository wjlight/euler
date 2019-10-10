# This file will be configured to contain variables for CPack. These variables
# should be set in the CMake list file of the project before CPack module is
# included. The list of available CPACK_xxx variables and their associated
# documentation may be obtained using
#  cpack --help-variable-list
#
# Some variables are common to all generators (e.g. CPACK_PACKAGE_NAME)
# and some are specific to a generator
# (e.g. CPACK_NSIS_EXTRA_INSTALL_COMMANDS). The generator specific variables
# usually begin with CPACK_<GENNAME>_xxxx.


SET(CPACK_BINARY_7Z "")
SET(CPACK_BINARY_BUNDLE "")
SET(CPACK_BINARY_CYGWIN "")
SET(CPACK_BINARY_DEB "")
SET(CPACK_BINARY_DRAGNDROP "")
SET(CPACK_BINARY_FREEBSD "")
SET(CPACK_BINARY_IFW "")
SET(CPACK_BINARY_NSIS "")
SET(CPACK_BINARY_OSXX11 "")
SET(CPACK_BINARY_PACKAGEMAKER "")
SET(CPACK_BINARY_PRODUCTBUILD "")
SET(CPACK_BINARY_RPM "")
SET(CPACK_BINARY_STGZ "")
SET(CPACK_BINARY_TBZ2 "")
SET(CPACK_BINARY_TGZ "")
SET(CPACK_BINARY_TXZ "")
SET(CPACK_BINARY_TZ "")
SET(CPACK_BINARY_WIX "")
SET(CPACK_BINARY_ZIP "")
SET(CPACK_BUILD_SOURCE_DIRS "/tmp/gflags;/tmp/gflags/build")
SET(CPACK_CMAKE_GENERATOR "Unix Makefiles")
SET(CPACK_COMPONENT_UNSPECIFIED_HIDDEN "TRUE")
SET(CPACK_COMPONENT_UNSPECIFIED_REQUIRED "TRUE")
SET(CPACK_GENERATOR "TGZ;ZIP")
SET(CPACK_GENERATOR "TGZ;ZIP")
SET(CPACK_IGNORE_FILES "/\\.git/;\\.swp$;\\.#;/#;\\.*~;cscope\\.*;/[Bb]uild[.+-_a-zA-Z0-9]*/")
SET(CPACK_INCLUDE_TOPLEVEL_DIRECTORY "TRUE")
SET(CPACK_INSTALLED_DIRECTORIES "/tmp/gflags;/")
SET(CPACK_INSTALL_CMAKE_PROJECTS "")
SET(CPACK_INSTALL_PREFIX "/usr/local")
SET(CPACK_MODULE_PATH "/tmp/gflags/cmake")
SET(CPACK_MONOLITHIC_INSTALL "TRUE")
SET(CPACK_NSIS_DISPLAY_NAME "gflags 2.2.0")
SET(CPACK_NSIS_INSTALLER_ICON_CODE "")
SET(CPACK_NSIS_INSTALLER_MUI_ICON_CODE "")
SET(CPACK_NSIS_INSTALL_ROOT "$PROGRAMFILES")
SET(CPACK_NSIS_PACKAGE_NAME "gflags 2.2.0")
SET(CPACK_OUTPUT_CONFIG_FILE "/tmp/gflags/build/CPackConfig.cmake")
SET(CPACK_OUTPUT_FILE_PREFIX "packages")
SET(CPACK_PACKAGE_ARCHITECTURE "amd64")
SET(CPACK_PACKAGE_CONTACT "google-gflags@googlegroups.com")
SET(CPACK_PACKAGE_DEFAULT_LOCATION "/")
SET(CPACK_PACKAGE_DESCRIPTION_FILE "/tmp/gflags/build/README.txt")
SET(CPACK_PACKAGE_DESCRIPTION_SUMMARY "A commandline flags library that allows for distributed flags.")
SET(CPACK_PACKAGE_FILE_NAME "gflags-2.2.0")
SET(CPACK_PACKAGE_INSTALL_DIRECTORY "gflags 2.2.0")
SET(CPACK_PACKAGE_INSTALL_REGISTRY_KEY "gflags 2.2.0")
SET(CPACK_PACKAGE_NAME "gflags")
SET(CPACK_PACKAGE_RELOCATABLE "true")
SET(CPACK_PACKAGE_VENDOR "Andreas Schuh")
SET(CPACK_PACKAGE_VERSION "2.2.0")
SET(CPACK_PACKAGE_VERSION_MAJOR "2")
SET(CPACK_PACKAGE_VERSION_MINOR "2")
SET(CPACK_PACKAGE_VERSION_PATCH "0")
SET(CPACK_PROJECT_CONFIG_FILE "/tmp/gflags/build/gflags-package.cmake")
SET(CPACK_RESOURCE_FILE_LICENSE "/tmp/gflags/COPYING.txt")
SET(CPACK_RESOURCE_FILE_README "/tmp/gflags/doc/index.html")
SET(CPACK_RESOURCE_FILE_WELCOME "/tmp/gflags/build/README.txt")
SET(CPACK_RPM_CHANGELOG_FILE "/tmp/gflags/ChangeLog.txt")
SET(CPACK_RPM_PACKAGE_GROUP "Development/Libraries")
SET(CPACK_RPM_PACKAGE_LICENSE "BSD")
SET(CPACK_RPM_PACKAGE_SOURCES "ON")
SET(CPACK_RPM_PACKAGE_URL "http://gflags.github.io/gflags")
SET(CPACK_SET_DESTDIR "OFF")
SET(CPACK_SOURCE_7Z "")
SET(CPACK_SOURCE_CYGWIN "")
SET(CPACK_SOURCE_GENERATOR "TGZ;ZIP")
SET(CPACK_SOURCE_IGNORE_FILES "/\\.git/;\\.swp$;\\.#;/#;\\.*~;cscope\\.*;/[Bb]uild[.+-_a-zA-Z0-9]*/")
SET(CPACK_SOURCE_INSTALLED_DIRECTORIES "/tmp/gflags;/")
SET(CPACK_SOURCE_OUTPUT_CONFIG_FILE "/tmp/gflags/build/CPackSourceConfig.cmake")
SET(CPACK_SOURCE_PACKAGE_FILE_NAME "gflags-2.2.0")
SET(CPACK_SOURCE_RPM "")
SET(CPACK_SOURCE_TBZ2 "")
SET(CPACK_SOURCE_TGZ "")
SET(CPACK_SOURCE_TOPLEVEL_TAG "source")
SET(CPACK_SOURCE_TXZ "")
SET(CPACK_SOURCE_TZ "")
SET(CPACK_SOURCE_ZIP "")
SET(CPACK_STRIP_FILES "")
SET(CPACK_SYSTEM_NAME "linux")
SET(CPACK_TOPLEVEL_TAG "source")
SET(CPACK_WIX_SIZEOF_VOID_P "8")

if(NOT CPACK_PROPERTIES_FILE)
  set(CPACK_PROPERTIES_FILE "/tmp/gflags/build/CPackProperties.cmake")
endif()

if(EXISTS ${CPACK_PROPERTIES_FILE})
  include(${CPACK_PROPERTIES_FILE})
endif()
