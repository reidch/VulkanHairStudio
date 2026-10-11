cmake_minimum_required(VERSION 3.24)
project(vulkan_hair_v47_directional_reflection LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Vulkan REQUIRED)
find_package(Threads REQUIRED)

include(FetchContent)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
set(JSON_BuildTests OFF CACHE BOOL "" FORCE)
set(JSON_Install OFF CACHE BOOL "" FORCE)
FetchContent_Declare(glfw GIT_REPOSITORY https://github.com/glfw/glfw.git GIT_TAG 3.4 GIT_SHALLOW TRUE)
FetchContent_Declare(glm GIT_REPOSITORY https://github.com/g-truc/glm.git GIT_TAG 1.0.1 GIT_SHALLOW TRUE)
FetchContent_Declare(imgui_src GIT_REPOSITORY https://github.com/ocornut/imgui.git GIT_TAG v1.91.8 GIT_SHALLOW TRUE)
FetchContent_Declare(nlohmann_json GIT_REPOSITORY https://github.com/nlohmann/json.git GIT_TAG v3.11.3 GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(glfw glm imgui_src nlohmann_json)

# -----------------------------------------------------------------------------
# Authoritative character asset: complete released ICT FaceKit Light neutral OBJ (MIT).
# Every source face is rendered and participates in contact. M_Face + M_BackHead
# alone define the external human solid sign; the remaining components are
# two-sided obstacles. The large morph-target repository is not vendored. For offline builds, point
# ICT_FACEKIT_LOCAL_OBJ / ICT_FACEKIT_LOCAL_LICENSE at local copies.
# -----------------------------------------------------------------------------
set(ICT_FACEKIT_LOCAL_OBJ "" CACHE FILEPATH "Optional local ICT FaceKit Light generic_neutral_mesh.obj")
set(ICT_FACEKIT_LOCAL_LICENSE "" CACHE FILEPATH "Optional local ICT FaceKit LICENSE")
set(ICT_ASSET_DIR "${CMAKE_CURRENT_BINARY_DIR}/ict_facekit")
file(MAKE_DIRECTORY "${ICT_ASSET_DIR}")
set(ICT_OBJ "${ICT_ASSET_DIR}/generic_neutral_mesh.obj")
set(ICT_LICENSE "${ICT_ASSET_DIR}/LICENSE")

if(ICT_FACEKIT_LOCAL_OBJ)
    configure_file("${ICT_FACEKIT_LOCAL_OBJ}" "${ICT_OBJ}" COPYONLY)
else()
    if(NOT EXISTS "${ICT_OBJ}")
        message(STATUS "Fetching ICT FaceKit Light neutral head (MIT, one-time configure download)...")
        file(DOWNLOAD
            "https://raw.githubusercontent.com/USC-ICT/ICT-FaceKit/master/FaceXModel/generic_neutral_mesh.obj"
            "${ICT_OBJ}"
            TLS_VERIFY ON
            STATUS ICT_DOWNLOAD_STATUS
            SHOW_PROGRESS)
        list(GET ICT_DOWNLOAD_STATUS 0 ICT_DOWNLOAD_CODE)
        if(NOT ICT_DOWNLOAD_CODE EQUAL 0)
            list(GET ICT_DOWNLOAD_STATUS 1 ICT_DOWNLOAD_MESSAGE)
            message(FATAL_ERROR "ICT FaceKit download failed: ${ICT_DOWNLOAD_MESSAGE}. Set -DICT_FACEKIT_LOCAL_OBJ=<path> for an offline build.")
        endif()
    endif()
endif()
file(SIZE "${ICT_OBJ}" ICT_OBJ_SIZE)
if(ICT_OBJ_SIZE LESS 1000000)
    message(FATAL_ERROR "ICT FaceKit neutral OBJ is unexpectedly small (${ICT_OBJ_SIZE} bytes). Remove it and reconfigure, or provide ICT_FACEKIT_LOCAL_OBJ.")
endif()

if(ICT_FACEKIT_LOCAL_LICENSE)
    configure_file("${ICT_FACEKIT_LOCAL_LICENSE}" "${ICT_LICENSE}" COPYONLY)
elseif(NOT EXISTS "${ICT_LICENSE}")
    file(DOWNLOAD
        "https://raw.githubusercontent.com/USC-ICT/ICT-FaceKit/master/LICENSE"
        "${ICT_LICENSE}"
        TLS_VERIFY ON
        STATUS ICT_LICENSE_STATUS)
    list(GET ICT_LICENSE_STATUS 0 ICT_LICENSE_CODE)
    if(NOT ICT_LICENSE_CODE EQUAL 0)
        message(FATAL_ERROR "Could not download the ICT FaceKit MIT license. Provide -DICT_FACEKIT_LOCAL_LICENSE=<path>.")
    endif()
endif()

add_library(imgui STATIC
    ${imgui_src_SOURCE_DIR}/imgui.cpp
    ${imgui_src_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_src_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_src_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_src_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${imgui_src_SOURCE_DIR}/backends/imgui_impl_vulkan.cpp
)
target_include_directories(imgui PUBLIC ${imgui_src_SOURCE_DIR} ${imgui_src_SOURCE_DIR}/backends)
target_link_libraries(imgui PUBLIC Vulkan::Vulkan glfw)

add_executable(vulkan_hair_v47
    src/main.cpp
    src/character_surface.cpp
    src/groom_binding.cpp
    src/groom_builder.cpp
    src/groom_cache.cpp
    src/groom_contact.cpp
    src/groom_style.cpp
    src/editor_ui.cpp
)
target_include_directories(vulkan_hair_v47 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_definitions(vulkan_hair_v47 PRIVATE GLM_FORCE_RADIANS GLM_FORCE_DEPTH_ZERO_TO_ONE)
target_link_libraries(vulkan_hair_v47 PRIVATE Vulkan::Vulkan glfw glm::glm imgui nlohmann_json::nlohmann_json
    Threads::Threads
)

if(MSVC)
    target_compile_options(vulkan_hair_v47 PRIVATE /W4 /permissive- /EHsc)
else()
    target_compile_options(vulkan_hair_v47 PRIVATE -Wall -Wextra -Wpedantic)
endif()

find_program(GLSLC_EXECUTABLE NAMES glslc HINTS "$ENV{VULKAN_SDK}/Bin" "$ENV{VULKAN_SDK}/bin")
if(NOT GLSLC_EXECUTABLE)
    message(FATAL_ERROR "glslc was not found. Install the Vulkan SDK and ensure VULKAN_SDK is set.")
endif()

set(SPV_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated/shaders")
file(MAKE_DIRECTORY "${SPV_DIR}")
function(compile_shader file stage)
    get_filename_component(name "${file}" NAME)
    set(out "${SPV_DIR}/${name}.spv")
    add_custom_command(
        OUTPUT "${out}"
        COMMAND "${GLSLC_EXECUTABLE}" --target-env=vulkan1.3 -fshader-stage=${stage}
                -I${CMAKE_CURRENT_SOURCE_DIR}/shaders
                "${CMAKE_CURRENT_SOURCE_DIR}/shaders/${file}" -o "${out}"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/${file}"
                "${CMAKE_CURRENT_SOURCE_DIR}/shaders/hair_physics_common.glsl"
                "${CMAKE_CURRENT_SOURCE_DIR}/shaders/hair_render_common.glsl"
                "${CMAKE_CURRENT_SOURCE_DIR}/shaders/hair_shadow_common.glsl"
                "${CMAKE_CURRENT_SOURCE_DIR}/shaders/hair_optical_sampling.glsl"
                "${CMAKE_CURRENT_SOURCE_DIR}/shaders/hair_bsdf.glsl"
                "${CMAKE_CURRENT_SOURCE_DIR}/shaders/hair_surface_brdf.glsl"
        COMMENT "Compiling ${file}"
        VERBATIM)
    set(SPV_FILES ${SPV_FILES} "${out}" PARENT_SCOPE)
endfunction()

compile_shader(head.vert vertex)
compile_shader(head.frag fragment)
compile_shader(sky.vert vertex)
compile_shader(sky.frag fragment)
compile_shader(hair.task task)
compile_shader(hair.mesh mesh)
compile_shader(hair.frag fragment)
compile_shader(hair_compose.frag fragment)
compile_shader(tonemap.vert vertex)
compile_shader(tonemap.frag fragment)
compile_shader(guide_sim.comp compute)
compile_shader(volume_clear.comp compute)
compile_shader(volume_p2g.comp compute)
compile_shader(volume_solve.comp compute)
compile_shader(volume_g2p.comp compute)
compile_shader(optical_splat.comp compute)
compile_shader(optical_depth.comp compute)
compile_shader(dom_splat.comp compute)
compile_shader(dom_prefix.comp compute)

add_custom_target(hair_shaders ALL DEPENDS ${SPV_FILES})
add_dependencies(vulkan_hair_v47 hair_shaders)

add_custom_command(TARGET vulkan_hair_v47 POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:vulkan_hair_v47>/shaders"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SPV_FILES} "$<TARGET_FILE_DIR:vulkan_hair_v47>/shaders"
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_CURRENT_SOURCE_DIR}/assets" "$<TARGET_FILE_DIR:vulkan_hair_v47>/assets"
    COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:vulkan_hair_v47>/assets/character/ict_facekit"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${ICT_OBJ}" "$<TARGET_FILE_DIR:vulkan_hair_v47>/assets/character/ict_facekit/generic_neutral_mesh.obj"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${ICT_LICENSE}" "$<TARGET_FILE_DIR:vulkan_hair_v47>/assets/character/ict_facekit/LICENSE"
)
