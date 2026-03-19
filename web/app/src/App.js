import { useEffect, useMemo, useRef } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { Grid, Line, OrbitControls, Text, Trail } from '@react-three/drei';
import { AxesHelper as ThreeAxesHelper } from 'three';
import './App.css';

const TOUCHX_X_MIN = -210;
const TOUCHX_X_MAX = 210;
const TOUCHX_Y_MIN = -100;
const TOUCHX_Y_MAX = 95;
const TOUCHX_Z_MIN = -145;
const TOUCHX_Z_MAX = 95;

const A = [
  [0, 0, 1],   // pb_x = touch_z
  [1, 0, 0],   // pb_y = touch_x
  [0, 1, 0],  // pb_z = -touch_y
];
const SCALE = 0.002;
const TOUCHX_CENTER = [0, 95, -110];
const VIEW_PADDING = 0.05;
const SIM_WS_URL = 'ws://127.0.0.1:8001/ws';

function touchxToPb(txPos) {
  const shifted = [
    txPos[0] - TOUCHX_CENTER[0],
    txPos[1] - TOUCHX_CENTER[1],
    txPos[2] - TOUCHX_CENTER[2],
  ];
  return A.map(
    (row) => SCALE * (row[0] * shifted[0] + row[1] * shifted[1] + row[2] * shifted[2]),
  );
}

function computePbBounds() {
  const xs = [TOUCHX_X_MIN, TOUCHX_X_MAX];
  const ys = [TOUCHX_Y_MIN, TOUCHX_Y_MAX];
  const zs = [TOUCHX_Z_MIN, TOUCHX_Z_MAX];
  const corners = [];

  xs.forEach((x) => {
    ys.forEach((y) => {
      zs.forEach((z) => {
        corners.push(touchxToPb([x, y, z]));
      });
    });
  });

  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  corners.forEach((corner) => {
    for (let i = 0; i < 3; i += 1) {
      min[i] = Math.min(min[i], corner[i]);
      max[i] = Math.max(max[i], corner[i]);
    }
  });
  return { min, max };
}

const { min: PB_FULL_MIN, max: PB_FULL_MAX } = computePbBounds();
const PB_VIEW_MIN = PB_FULL_MIN.map((v) => v - VIEW_PADDING);
const PB_VIEW_MAX = PB_FULL_MAX.map((v) => v + VIEW_PADDING);

export function useSimSocket(url) {
  const wsRef = useRef(null);
  const latestPositionRef = useRef(null);

  useEffect(() => {
    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => {
      ws.send('connect');
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (
          data
          && Number.isFinite(data.x)
          && Number.isFinite(data.y)
          && Number.isFinite(data.z)
        ) {
          latestPositionRef.current = [data.x, data.y, data.z];
        }
      } catch (_err) {
        // Ignore malformed payloads and keep last known good position.
      }
    };

    return () => {
      ws.close();
      wsRef.current = null;
    };
  }, [url]);

  return { latestPositionRef, wsRef };
}

function getBoxVertices(mins, maxs) {
  return [
    [mins[0], mins[1], mins[2]],
    [mins[0], mins[1], maxs[2]],
    [mins[0], maxs[1], mins[2]],
    [mins[0], maxs[1], maxs[2]],
    [maxs[0], mins[1], mins[2]],
    [maxs[0], mins[1], maxs[2]],
    [maxs[0], maxs[1], mins[2]],
    [maxs[0], maxs[1], maxs[2]],
  ];
}

function BoundingBoxGraph() {
  const vertices = getBoxVertices(PB_FULL_MIN, PB_FULL_MAX);
  const edgeIndices = [
    [0, 1], [0, 2], [0, 4],
    [1, 3], [1, 5],
    [2, 3], [2, 6],
    [3, 7],
    [4, 5], [4, 6],
    [5, 7],
    [6, 7],
  ];

  return (
    <group>
      {edgeIndices.map(([from, to], idx) => (
        <Line
          key={`edge-${idx}`}
          points={[vertices[from], vertices[to]]}
          color="#4f8cff"
          lineWidth={1.5}
        />
      ))}
    </group>
  );
}

function AxisLabels({ boundsMax }) {
  const xLabelPos = boundsMax[0] + 0.06;
  const yLabelPos = boundsMax[1] + 0.06;
  const zLabelPos = boundsMax[2] + 0.06;

  return (
    <group>
      <Text position={[xLabelPos, 0, 0]} fontSize={0.04} color="#ff5f5f" anchorX="center" anchorY="middle">
        X
      </Text>
      <Text position={[0, yLabelPos, 0]} fontSize={0.04} color="#5fff7a" anchorX="center" anchorY="middle">
        Y
      </Text>
      <Text position={[0, 0, zLabelPos]} fontSize={0.04} color="#5fb4ff" anchorX="center" anchorY="middle">
        Z
      </Text>
    </group>
  );
}

function RobotMarker({ latestPositionRef, initialPosition }) {
  const markerRef = useRef(null);

  useFrame(() => {
    if (!markerRef.current || !latestPositionRef.current) {
      return;
    }
    const [x, y, z] = latestPositionRef.current;
    markerRef.current.position.set(x, y, z);
  });

  return (
    <Trail
      width={0.01}
      length={0.35}
      color="#ff7b7b"
      attenuation={(t) => t * t}
    >
      <mesh ref={markerRef} position={initialPosition}>
        <sphereGeometry args={[0.012, 24, 24]} />
        <meshStandardMaterial color="#ff2d2d" />
      </mesh>
    </Trail>
  );
}

function App() {
  const { latestPositionRef } = useSimSocket(SIM_WS_URL);
  const center = useMemo(
    () => [
      (PB_FULL_MIN[0] + PB_FULL_MAX[0]) / 2,
      (PB_FULL_MIN[1] + PB_FULL_MAX[1]) / 2,
      (PB_FULL_MIN[2] + PB_FULL_MAX[2]) / 2,
    ],
    [],
  );
  const cameraPosition = useMemo(
    () => [
      PB_VIEW_MAX[0] + 0.45,
      PB_VIEW_MAX[1] + 0.45,
      PB_VIEW_MAX[2] + 0.45,
    ],
    [],
  );
  const axesSize = useMemo(() => Math.max(...PB_VIEW_MAX.map((v) => Math.abs(v))) + 0.1, []);
  const gridSize = useMemo(
    () => Math.max(PB_VIEW_MAX[0] - PB_VIEW_MIN[0], PB_VIEW_MAX[1] - PB_VIEW_MIN[1]) + 0.5,
    [],
  );

  return (
    <div className="app">
      <Canvas camera={{ position: cameraPosition, fov: 50, up: [0, 0, 1] }}>
        <color attach="background" args={['#ffffff']} />
        <ambientLight intensity={0.5} />
        <directionalLight position={[2, 2, 2]} intensity={1} />
        <Grid
          position={[0, 0, PB_VIEW_MIN[2]]}
          rotation={[Math.PI / 2, 0, 0]}
          args={[gridSize, gridSize]}
          cellSize={0.05}
          sectionSize={0.25}
          cellColor="#dddddd"
          sectionColor="#d3d3d3"
          fadeDistance={3}
          fadeStrength={1}
        />

        <primitive object={new ThreeAxesHelper(axesSize)} />
        <AxisLabels boundsMax={PB_VIEW_MAX} />
        <BoundingBoxGraph />
        <RobotMarker latestPositionRef={latestPositionRef} initialPosition={center} />

        <OrbitControls makeDefault target={center} />
      </Canvas>
    </div>
  );
}

export default App;
