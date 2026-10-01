"use client";

import { useEffect, useRef } from "react";

import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { STLLoader } from "three/examples/jsm/loaders/STLLoader.js";


type StlViewerProps = {
    src: string;
    height?: number;
};


export default function StlViewer({
    src,
    height = 360
}: StlViewerProps) {

    const containerRef =
        useRef<HTMLDivElement | null>(null);


    useEffect(() => {

        const container =
            containerRef.current;

        if (!container || !src) {
            return;
        }


        /* =====================================================
           SCENE
        ===================================================== */

        const scene =
            new THREE.Scene();

        scene.background =
            new THREE.Color(0xf4f4f2);


        /* =====================================================
           CAMERA
        ===================================================== */

        const camera =
            new THREE.PerspectiveCamera(
                45,
                1,
                0.1,
                100000
            );


        /* =====================================================
           RENDERER
        ===================================================== */

        const renderer =
            new THREE.WebGLRenderer({
                antialias: true
            });

        renderer.setPixelRatio(
            Math.min(
                window.devicePixelRatio,
                2
            )
        );

        renderer.outputColorSpace =
            THREE.SRGBColorSpace;

        renderer.shadowMap.enabled =
            true;


        container.appendChild(
            renderer.domElement
        );


        /* =====================================================
           CONTROLS
        ===================================================== */

        const controls =
            new OrbitControls(
                camera,
                renderer.domElement
            );

        controls.enableDamping =
            true;

        controls.dampingFactor =
            0.08;


        /* =====================================================
           LIGHTS
        ===================================================== */

        const ambientLight =
            new THREE.AmbientLight(
                0xffffff,
                1.6
            );

        scene.add(
            ambientLight
        );


        const keyLight =
            new THREE.DirectionalLight(
                0xffffff,
                2.4
            );

        keyLight.position.set(
            3,
            4,
            5
        );

        scene.add(
            keyLight
        );


        const fillLight =
            new THREE.DirectionalLight(
                0xffffff,
                1.0
            );

        fillLight.position.set(
            -4,
            -2,
            3
        );

        scene.add(
            fillLight
        );


        /* =====================================================
           MODEL
        ===================================================== */

        let mesh:
            THREE.Mesh<
                THREE.BufferGeometry,
                THREE.MeshStandardMaterial
            >
            | null = null;


        const loader =
            new STLLoader();


        loader.load(

            src,

            (geometry) => {

                geometry.computeVertexNormals();
                geometry.computeBoundingBox();


                const material =
                    new THREE.MeshStandardMaterial({
                        color: 0xb9b9b4,
                        roughness: 0.72,
                        metalness: 0.05,
                        side: THREE.DoubleSide
                    });


                mesh =
                    new THREE.Mesh(
                        geometry,
                        material
                    );


                /* -------------------------------------------------
                   CENTER MODEL
                ------------------------------------------------- */

                const box =
                    new THREE.Box3().setFromObject(
                        mesh
                    );


                const center =
                    box.getCenter(
                        new THREE.Vector3()
                    );


                mesh.position.sub(
                    center
                );


                scene.add(
                    mesh
                );


                /* -------------------------------------------------
                   FIT CAMERA TO MODEL
                ------------------------------------------------- */

                const centeredBox =
                    new THREE.Box3().setFromObject(
                        mesh
                    );


                const size =
                    centeredBox.getSize(
                        new THREE.Vector3()
                    );


                const maxDimension =
                    Math.max(
                        size.x,
                        size.y,
                        size.z
                    );


                const fovRadians =
                    camera.fov *
                    Math.PI /
                    180;


                const distance =
                    (
                        maxDimension /
                        2
                    ) /
                    Math.tan(
                        fovRadians /
                        2
                    ) *
                    1.65;


                camera.position.set(
                    distance,
                    distance,
                    distance
                );


                camera.near =
                    Math.max(
                        distance / 1000,
                        0.01
                    );


                camera.far =
                    Math.max(
                        distance * 100,
                        1000
                    );


                camera.updateProjectionMatrix();


                controls.target.set(
                    0,
                    0,
                    0
                );

                controls.update();

            },

            undefined,

            (error) => {

                console.error(
                    "Could not load STL:",
                    error
                );

            }

        );


        /* =====================================================
           RESIZE
        ===================================================== */

        const resize = () => {

            const width =
                container.clientWidth;

            const currentHeight =
                container.clientHeight;


            if (
                width <= 0 ||
                currentHeight <= 0
            ) {
                return;
            }


            renderer.setSize(
                width,
                currentHeight,
                false
            );


            camera.aspect =
                width /
                currentHeight;


            camera.updateProjectionMatrix();

        };


        const resizeObserver =
            new ResizeObserver(
                resize
            );


        resizeObserver.observe(
            container
        );

        resize();


        /* =====================================================
           RENDER LOOP
        ===================================================== */

        let animationFrame = 0;


        const animate = () => {

            controls.update();

            renderer.render(
                scene,
                camera
            );

            animationFrame =
                requestAnimationFrame(
                    animate
                );

        };


        animate();


        /* =====================================================
           CLEANUP
        ===================================================== */

        return () => {

            cancelAnimationFrame(
                animationFrame
            );


            resizeObserver.disconnect();

            controls.dispose();


            if (mesh) {

                mesh.geometry.dispose();

                mesh.material.dispose();

                scene.remove(
                    mesh
                );

            }


            renderer.dispose();


            if (
                renderer.domElement.parentElement
                === container
            ) {

                container.removeChild(
                    renderer.domElement
                );

            }

        };

    }, [src]);


    return (

        <div
            ref={containerRef}

            style={{
                width: "100%",
                height: `${height}px`,
                border: "1px solid #d5d5d0",
                borderRadius: "8px",
                overflow: "hidden",
                background: "#f4f4f2"
            }}
        />

    );

}
