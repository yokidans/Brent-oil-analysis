// src/dashboard/frontend/src/components/NeuralCPVisualization.js
import React, { useRef, useEffect } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import { useFetch } from '../hooks/useFetch';

const NeuralCPVisualization = () => {
    const mountRef = useRef(null);
    const { data, loading } = useFetch('/neural_cp/visualization');
    
    useEffect(() => {
        if (loading || !data) return;
        
        // Initialize Three.js scene
        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        const renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setSize(mountRef.current.clientWidth, mountRef.current.clientHeight);
        mountRef.current.appendChild(renderer.domElement);
        
        // Add controls
        const controls = new OrbitControls(camera, renderer.domElement);
        
        // Create latent space visualization
        const latentSpace = data.latent_space;
        const geometry = new THREE.BufferGeometry();
        const vertices = new Float32Array(latentSpace.flat());
        geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
        
        const material = new THREE.PointsMaterial({
            color: 0x8888ff,
            size: 0.1,
            transparent: true,
            opacity: 0.8
        });
        
        const points = new THREE.Points(geometry, material);
        scene.add(points);
        
        // Add change point markers
        data.changepoints.forEach((cp, i) => {
            const sphereGeometry = new THREE.SphereGeometry(0.2);
            const sphereMaterial = new THREE.MeshBasicMaterial({
                color: new THREE.Color(`hsl(${i * 60}, 100%, 50%)`),
                transparent: true,
                opacity: 0.7
            });
            const sphere = new THREE.Mesh(sphereGeometry, sphereMaterial);
            sphere.position.set(
                latentSpace[i][0] * 2,
                latentSpace[i][1] * 2,
                latentSpace[i][2] * 2
            );
            scene.add(sphere);
        });
        
        // Add attention flow lines
        const lineMaterial = new THREE.LineBasicMaterial({ color: 0xffffff });
        data.attention.forEach((weights, i) => {
            weights.forEach((w, j) => {
                if (w > 0.3) {
                    const points = [
                        new THREE.Vector3(
                            latentSpace[i][0] * 2,
                            latentSpace[i][1] * 2,
                            latentSpace[i][2] * 2
                        ),
                        new THREE.Vector3(
                            latentSpace[j][0] * 2,
                            latentSpace[j][1] * 2,
                            latentSpace[j][2] * 2
                        )
                    ];
                    const lineGeometry = new THREE.BufferGeometry().setFromPoints(points);
                    const line = new THREE.Line(lineGeometry, lineMaterial);
                    scene.add(line);
                }
            });
        });
        
        camera.position.z = 5;
        
        const animate = () => {
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        };
        
        animate();
        
        return () => {
            mountRef.current.removeChild(renderer.domElement);
        };
    }, [data, loading]);
    
    return (
        <div 
            ref={mountRef} 
            style={{ 
                width: '100%', 
                height: '600px',
                border: '1px solid #333',
                borderRadius: '8px'
            }}
        />
    );
};

export default NeuralCPVisualization;