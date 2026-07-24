"use client";

import { useState, useEffect, useRef } from "react";
import { startTraining, stopTraining, getTrainingStatus, getTrainingRuns } from "@/lib/api";
import {
  Play,
  Square,
  Activity,
  Zap,
  TrendingDown,
} from "lucide-react";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from "chart.js";
import { Line } from "react-chartjs-2";

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

export default function TrainingPage() {
  const [isTraining, setIsTraining] = useState(false);
  const [currentRun, setCurrentRun] = useState(null);
  const [epochs, setEpochs] = useState([]);
  const [runs, setRuns] = useState([]);
  const [error, setError] = useState(null);
  const [actionLoading, setActionLoading] = useState(false);

  const wsRef = useRef(null);

  // Connect WebSocket for live updates
  useEffect(() => {
    function connectWs() {
      const ws = new WebSocket("ws://localhost:8000/ws/training");
      wsRef.current = ws;

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.type === "training_update") {
            setIsTraining(true);
            setCurrentRun({
              id: data.run_id,
              status: data.status,
              completed_epochs: data.completed_epochs,
              total_epochs: data.total_epochs,
              best_epoch: data.best_epoch,
              best_val_loss: data.best_val_loss,
            });
            setEpochs(data.epochs || []);
          } else if (data.type === "training_complete") {
            setIsTraining(false);
            loadRuns();
          } else if (data.type === "no_training") {
            setIsTraining(false);
          }
        } catch (err) {
          console.error("WS Parse Error:", err);
        }
      };

      ws.onclose = () => {
        // Reconnect after 3 seconds if closed
        setTimeout(connectWs, 3000);
      };
    }

    connectWs();
    loadRuns();

    return () => {
      if (wsRef.current) wsRef.current.close();
    };
  }, []);

  async function loadRuns() {
    try {
      const data = await getTrainingRuns();
      setRuns(data || []);
    } catch (err) {
      console.error(err);
    }
  }

  const handleStartTraining = async () => {
    setActionLoading(true);
    setError(null);
    try {
      await startTraining();
      setIsTraining(true);
    } catch (err) {
      setError(err.message || "Failed to start training");
    } finally {
      setActionLoading(false);
    }
  };

  const handleStopTraining = async () => {
    setActionLoading(true);
    try {
      await stopTraining();
    } catch (err) {
      setError(err.message || "Failed to stop training");
    } finally {
      setActionLoading(false);
    }
  };

  // Chart options & data
  const epochLabels = epochs.map((e) => `Epoch ${e.epoch}`);
  const lossChartData = {
    labels: epochLabels,
    datasets: [
      {
        label: "Train Loss",
        data: epochs.map((e) => e.train_loss),
        borderColor: "#06b6d4",
        backgroundColor: "rgba(6, 182, 212, 0.1)",
        tension: 0.3,
      },
      {
        label: "Validation Loss",
        data: epochs.map((e) => e.val_loss),
        borderColor: "#a855f7",
        backgroundColor: "rgba(168, 85, 247, 0.1)",
        tension: 0.3,
      },
    ],
  };

  const maeChartData = {
    labels: epochLabels,
    datasets: [
      {
        label: "Train MAE (min)",
        data: epochs.map((e) => e.train_mae),
        borderColor: "#10b981",
        tension: 0.3,
      },
      {
        label: "Validation MAE (min)",
        data: epochs.map((e) => e.val_mae),
        borderColor: "#f59e0b",
        tension: 0.3,
      },
    ],
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: "top",
        labels: { color: "#94a3b8", font: { size: 11 } },
      },
    },
    scales: {
      x: { grid: { color: "rgba(255,255,255,0.05)" }, ticks: { color: "#64748b" } },
      y: { grid: { color: "rgba(255,255,255,0.05)" }, ticks: { color: "#64748b" } },
    },
  };

  return (
    <div className="space-y-6">
      {/* Header & Controls */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-white tracking-tight">Live Training Monitor</h1>
          <p className="text-sm text-slate-400 mt-1">
            Real-time WebSocket monitoring of PyTorch model training across epochs
          </p>
        </div>

        <div className="flex items-center space-x-3">
          {isTraining ? (
            <button
              onClick={handleStopTraining}
              disabled={actionLoading}
              className="px-4 py-2.5 bg-rose-500/20 hover:bg-rose-500/30 text-rose-300 border border-rose-500/40 rounded-xl text-xs font-semibold flex items-center space-x-2 transition-all shadow-lg shadow-rose-500/10"
            >
              <Square className="w-4 h-4" />
              <span>Stop Training</span>
            </button>
          ) : (
            <button
              onClick={handleStartTraining}
              disabled={actionLoading}
              className="px-5 py-2.5 bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-bold rounded-xl text-xs flex items-center space-x-2 transition-all shadow-lg shadow-cyan-500/20"
            >
              <Play className="w-4 h-4 fill-slate-950" />
              <span>{actionLoading ? "Starting..." : "Start Training"}</span>
            </button>
          )}
        </div>
      </div>

      {error && (
        <div className="p-4 bg-rose-500/10 border border-rose-500/30 rounded-xl text-rose-300 text-sm">
          {error}
        </div>
      )}

      {/* Live Status Card */}
      <div className="glass-card p-6 border-l-4 border-l-cyan-500">
        <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
          <div className="flex items-center space-x-4">
            <div className={`p-3 rounded-xl ${isTraining ? "bg-cyan-500/20 text-cyan-400 animate-pulse" : "bg-slate-800 text-slate-400"}`}>
              <Activity className="w-6 h-6" />
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <h2 className="text-lg font-bold text-white">
                  {isTraining ? "Training Loop Running" : "Status: Idle"}
                </h2>
                <span className={`px-2.5 py-0.5 rounded-full text-[10px] font-bold uppercase ${isTraining ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/30" : "bg-slate-800 text-slate-400"
                  }`}>
                  {isTraining ? "Active" : "Ready"}
                </span>
              </div>
              <p className="text-xs text-slate-400 mt-1">
                {isTraining
                  ? `Training Run #${currentRun?.id || ""} • ${currentRun?.completed_epochs || 0} / ${currentRun?.total_epochs || 100} Epochs`
                  : "Click 'Start Training' to train model on 100% PostgreSQL dataset (85% train / 15% val split)"}
              </p>
            </div>
          </div>

          {isTraining && currentRun && (
            <div className="flex items-center space-x-6 text-xs font-mono">
              <div>
                <span className="text-slate-400 block">Best Epoch</span>
                <span className="text-emerald-400 font-bold text-base">#{currentRun.best_epoch || 1}</span>
              </div>
              <div>
                <span className="text-slate-400 block">Best Val Loss</span>
                <span className="text-purple-400 font-bold text-base">{currentRun.best_val_loss?.toFixed(4) || "-"}</span>
              </div>
            </div>
          )}
        </div>

        {/* Progress Bar */}
        {isTraining && currentRun && (
          <div className="mt-4 pt-4 border-t border-slate-800">
            <div className="flex justify-between text-xs text-slate-400 mb-1.5 font-mono">
              <span>Progress</span>
              <span>{Math.round(((currentRun.completed_epochs || 0) / (currentRun.total_epochs || 100)) * 100)}%</span>
            </div>
            <div className="w-full bg-slate-900 rounded-full h-2 overflow-hidden">
              <div
                className="bg-gradient-to-r from-cyan-500 to-purple-500 h-full transition-all duration-300"
                style={{
                  width: `${((currentRun.completed_epochs || 0) / (currentRun.total_epochs || 100)) * 100}%`,
                }}
              ></div>
            </div>
          </div>
        )}
      </div>

      {/* Live Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Loss Curve */}
        <div className="glass-card p-5">
          <h3 className="text-sm font-bold text-white mb-3 flex items-center space-x-2">
            <TrendingDown className="w-4 h-4 text-cyan-400" />
            <span>Loss Curves (Train vs Validation)</span>
          </h3>
          <div className="h-64">
            {epochs.length > 0 ? (
              <Line data={lossChartData} options={chartOptions} />
            ) : (
              <div className="h-full flex items-center justify-center text-xs text-slate-500">
                Start training to view real-time loss curves
              </div>
            )}
          </div>
        </div>

        {/* MAE Curve */}
        <div className="glass-card p-5">
          <h3 className="text-sm font-bold text-white mb-3 flex items-center space-x-2">
            <Zap className="w-4 h-4 text-emerald-400" />
            <span>ETA Error (MAE) per Epoch</span>
          </h3>
          <div className="h-64">
            {epochs.length > 0 ? (
              <Line data={maeChartData} options={chartOptions} />
            ) : (
              <div className="h-full flex items-center justify-center text-xs text-slate-500">
                Start training to view real-time MAE progress
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Epoch History Log */}
      {epochs.length > 0 && (
        <div className="glass-card overflow-hidden">
          <div className="p-4 border-b border-slate-800">
            <h3 className="text-sm font-bold text-white">Live Epoch Logs</h3>
          </div>
          <div className="max-h-60 overflow-y-auto">
            <table className="w-full text-left text-xs text-slate-300">
              <thead className="bg-slate-900/90 uppercase text-[11px] text-slate-400 sticky top-0">
                <tr>
                  <th className="px-4 py-2.5">Epoch</th>
                  <th className="px-4 py-2.5">Train Loss</th>
                  <th className="px-4 py-2.5">Val Loss</th>
                  <th className="px-4 py-2.5">Train MAE</th>
                  <th className="px-4 py-2.5">Val MAE</th>
                  <th className="px-4 py-2.5">Train F1</th>
                  <th className="px-4 py-2.5">Val F1</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60 font-mono">
                {epochs.map((ep) => (
                  <tr key={ep.epoch} className="hover:bg-slate-800/40">
                    <td className="px-4 py-2 font-bold text-cyan-400">#{ep.epoch}</td>
                    <td className="px-4 py-2">{ep.train_loss?.toFixed(4)}</td>
                    <td className="px-4 py-2">{ep.val_loss?.toFixed(4)}</td>
                    <td className="px-4 py-2">{ep.train_mae?.toFixed(2)} min</td>
                    <td className="px-4 py-2 text-emerald-400">{ep.val_mae?.toFixed(2)} min</td>
                    <td className="px-4 py-2">{(ep.train_f1 * 100).toFixed(1)}%</td>
                    <td className="px-4 py-2 text-purple-400">{(ep.val_f1 * 100).toFixed(1)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
