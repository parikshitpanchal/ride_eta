"use client";

import { useState, useEffect } from "react";
import {
  getPipelineStatus,
  getConfig,
  updateConfig,
  uploadCsv,
  runFeatureEngineering,
  runPrediction,
} from "@/lib/api";
import {
  UploadCloud,
  Cpu,
  PlayCircle,
  Save,
  Database,
  Sliders,
  CheckCircle2,
  AlertCircle,
  FileText,
} from "lucide-react";

export default function AdminPage() {
  const [pipelineStatus, setPipelineStatus] = useState(null);
  const [config, setConfigData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [message, setMessage] = useState(null);
  const [error, setError] = useState(null);

  // File Upload states
  const [rawFile, setRawFile] = useState(null);
  const [testFile, setTestFile] = useState(null);
  const [uploadingRaw, setUploadingRaw] = useState(false);
  const [runningFE, setRunningFE] = useState(false);
  const [runningPred, setRunningPred] = useState(false);

  // Editable hyperparams
  const [delayThreshold, setDelayThreshold] = useState(0.3);
  const [learningRate, setLearningRate] = useState(0.001);
  const [epochs, setEpochs] = useState(100);
  const [batchSize, setBatchSize] = useState(256);
  const [savingConfig, setSavingConfig] = useState(false);

  useEffect(() => {
    loadAdminData();
  }, []);

  async function loadAdminData() {
    setLoading(true);
    try {
      const [statusData, configData] = await Promise.all([
        getPipelineStatus(),
        getConfig(),
      ]);
      setPipelineStatus(statusData);
      setConfigData(configData);
      if (configData) {
        setDelayThreshold(configData.delay_threshold);
        setLearningRate(configData.learning_rate);
        setEpochs(configData.epochs);
        setBatchSize(configData.batch_size);
      }
    } catch (err) {
      setError(err.message || "Failed to load admin data");
    } finally {
      setLoading(false);
    }
  }

  const handleUploadRawCsv = async () => {
    if (!rawFile) return;
    setUploadingRaw(true);
    setMessage(null);
    setError(null);
    try {
      const res = await uploadCsv(rawFile);
      setMessage(`✅ ${res.message} (${res.rows_inserted} rows added)`);
      setRawFile(null);
      loadAdminData();
    } catch (err) {
      setError(err.message || "CSV upload failed");
    } finally {
      setUploadingRaw(false);
    }
  };

  const handleRunFeatureEngineering = async () => {
    setRunningFE(true);
    setMessage(null);
    setError(null);
    try {
      const res = await runFeatureEngineering();
      setMessage(`✅ ${res.message} (${res.rows_processed} orders engineered)`);
      loadAdminData();
    } catch (err) {
      setError(err.message || "Feature engineering failed");
    } finally {
      setRunningFE(false);
    }
  };

  const handleRunPrediction = async () => {
    if (!testFile) return;
    setRunningPred(true);
    setMessage(null);
    setError(null);
    try {
      const res = await runPrediction(testFile);
      setMessage(`✅ ${res.message}`);
      setTestFile(null);
      loadAdminData();
    } catch (err) {
      setError(err.message || "Prediction execution failed");
    } finally {
      setRunningPred(false);
    }
  };

  const handleSaveConfig = async (e) => {
    e.preventDefault();
    setSavingConfig(true);
    setMessage(null);
    setError(null);
    try {
      await updateConfig({
        delay_threshold: Number(delayThreshold),
        learning_rate: Number(learningRate),
        epochs: Number(epochs),
        batch_size: Number(batchSize),
      });
      setMessage("✅ Configuration updated successfully");
      loadAdminData();
    } catch (err) {
      setError(err.message || "Failed to update configuration");
    } finally {
      setSavingConfig(false);
    }
  };

  if (loading) {
    return (
      <div className="space-y-6 animate-pulse">
        <div className="h-8 bg-slate-800 rounded w-64"></div>
        <div className="h-48 bg-slate-900 rounded-xl"></div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-white tracking-tight">Admin & Data Pipeline Control</h1>
        <p className="text-sm text-slate-400 mt-1">
          Upload raw data, trigger feature engineering, run test predictions, and tune model hyperparameters
        </p>
      </div>

      {message && (
        <div className="p-4 bg-emerald-500/10 border border-emerald-500/30 rounded-xl text-emerald-300 text-sm flex items-center space-x-2">
          <CheckCircle2 className="w-4 h-4 text-emerald-400" />
          <span>{message}</span>
        </div>
      )}

      {error && (
        <div className="p-4 bg-rose-500/10 border border-rose-500/30 rounded-xl text-rose-300 text-sm flex items-center space-x-2">
          <AlertCircle className="w-4 h-4 text-rose-400" />
          <span>{error}</span>
        </div>
      )}

      {/* Pipeline Status Cards */}
      {pipelineStatus && (
        <div className="grid grid-cols-2 md:grid-cols-6 gap-4">
          <div className="glass-card p-4">
            <span className="text-xs text-slate-400 font-semibold uppercase">Raw Orders</span>
            <p className="text-xl font-bold text-white mt-1">{pipelineStatus.total_raw_orders.toLocaleString()}</p>
          </div>
          <div className="glass-card p-4">
            <span className="text-xs text-slate-400 font-semibold uppercase">Engineered</span>
            <p className="text-xl font-bold text-cyan-400 mt-1">{pipelineStatus.engineered_orders.toLocaleString()}</p>
          </div>
          <div className="glass-card p-4">
            <span className="text-xs text-slate-400 font-semibold uppercase">Unprocessed</span>
            <p className="text-xl font-bold text-amber-400 mt-1">{pipelineStatus.unprocessed_orders.toLocaleString()}</p>
          </div>
          <div className="glass-card p-4">
            <span className="text-xs text-slate-400 font-semibold uppercase">Predictions</span>
            <p className="text-xl font-bold text-purple-400 mt-1">{pipelineStatus.total_predictions.toLocaleString()}</p>
          </div>
          <div className="glass-card p-4">
            <span className="text-xs text-slate-400 font-semibold uppercase">Training Runs</span>
            <p className="text-xl font-bold text-emerald-400 mt-1">{pipelineStatus.total_training_runs}</p>
          </div>
          <div className="glass-card p-4">
            <span className="text-xs text-slate-400 font-semibold uppercase">Drivers</span>
            <p className="text-xl font-bold text-slate-200 mt-1">{pipelineStatus.total_drivers.toLocaleString()}</p>
          </div>
        </div>
      )}

      {/* Pipeline Control Actions */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Step 1: Upload Raw CSV */}
        <div className="glass-card p-6 flex flex-col justify-between">
          <div>
            <div className="flex items-center space-x-2 text-cyan-400 mb-3">
              <UploadCloud className="w-5 h-5" />
              <h3 className="font-bold text-white text-sm">1. Upload Raw Ride Orders</h3>
            </div>
            <p className="text-xs text-slate-400 mb-4">
              Upload production CSV file. Rows are saved into `raw_ride_orders` in PostgreSQL.
            </p>

            <div className="border-2 border-dashed border-slate-700/80 hover:border-cyan-500/50 rounded-xl p-4 text-center transition-colors">
              <input
                type="file"
                accept=".csv"
                id="raw-csv"
                onChange={(e) => setRawFile(e.target.files[0])}
                className="hidden"
              />
              <label htmlFor="raw-csv" className="cursor-pointer block">
                <FileText className="w-8 h-8 text-slate-500 mx-auto mb-2" />
                <span className="text-xs text-slate-300 font-medium block">
                  {rawFile ? rawFile.name : "Choose CSV file"}
                </span>
                <span className="text-[11px] text-slate-500 block mt-1">Click to browse</span>
              </label>
            </div>
          </div>

          <button
            onClick={handleUploadRawCsv}
            disabled={!rawFile || uploadingRaw}
            className="mt-4 w-full py-2.5 bg-cyan-500 hover:bg-cyan-400 disabled:opacity-40 disabled:cursor-not-allowed text-slate-950 font-bold rounded-xl text-xs transition-colors"
          >
            {uploadingRaw ? "Uploading..." : "Upload to Database"}
          </button>
        </div>

        {/* Step 2: Feature Engineering */}
        <div className="glass-card p-6 flex flex-col justify-between">
          <div>
            <div className="flex items-center space-x-2 text-purple-400 mb-3">
              <Cpu className="w-5 h-5" />
              <h3 className="font-bold text-white text-sm">2. Run Feature Engineering</h3>
            </div>
            <p className="text-xs text-slate-400 mb-4">
              Processes unprocessed orders in `raw_ride_orders` using `feature_engineering.py` and stores in `engineered_ride_orders`.
            </p>

            <div className="p-4 bg-slate-900/60 border border-slate-800 rounded-xl text-xs space-y-2 text-slate-400">
              <div className="flex justify-between">
                <span>Unprocessed Queue:</span>
                <span className="font-bold text-amber-400">{pipelineStatus?.unprocessed_orders || 0} orders</span>
              </div>
              <div className="flex justify-between">
                <span>Engineered Total:</span>
                <span className="font-bold text-cyan-400">{pipelineStatus?.engineered_orders || 0} orders</span>
              </div>
            </div>
          </div>

          <button
            onClick={handleRunFeatureEngineering}
            disabled={runningFE || pipelineStatus?.unprocessed_orders === 0}
            className="mt-4 w-full py-2.5 bg-purple-600 hover:bg-purple-500 disabled:opacity-40 disabled:cursor-not-allowed text-white font-bold rounded-xl text-xs transition-colors"
          >
            {runningFE ? "Engineering Features..." : "Run Feature Engineering"}
          </button>
        </div>

        {/* Step 3: Run Predictions on Test CSV */}
        <div className="glass-card p-6 flex flex-col justify-between">
          <div>
            <div className="flex items-center space-x-2 text-emerald-400 mb-3">
              <PlayCircle className="w-5 h-5" />
              <h3 className="font-bold text-white text-sm">3. Run Predictions on Test CSV</h3>
            </div>
            <p className="text-xs text-slate-400 mb-4">
              Upload a test CSV. Runs PyTorch predictor and saves predicted ETA + delay probability into `predictions`.
            </p>

            <div className="border-2 border-dashed border-slate-700/80 hover:border-emerald-500/50 rounded-xl p-4 text-center transition-colors">
              <input
                type="file"
                accept=".csv"
                id="test-csv"
                onChange={(e) => setTestFile(e.target.files[0])}
                className="hidden"
              />
              <label htmlFor="test-csv" className="cursor-pointer block">
                <FileText className="w-8 h-8 text-slate-500 mx-auto mb-2" />
                <span className="text-xs text-slate-300 font-medium block">
                  {testFile ? testFile.name : "Choose Test CSV file"}
                </span>
                <span className="text-[11px] text-slate-500 block mt-1">Option B: Upload test dataset</span>
              </label>
            </div>
          </div>

          <button
            onClick={handleRunPrediction}
            disabled={!testFile || runningPred}
            className="mt-4 w-full py-2.5 bg-emerald-500 hover:bg-emerald-400 disabled:opacity-40 disabled:cursor-not-allowed text-slate-950 font-bold rounded-xl text-xs transition-colors"
          >
            {runningPred ? "Running Predictor..." : "Execute Model Prediction"}
          </button>
        </div>
      </div>

      {/* Hyperparameter Config Editor */}
      <div className="glass-card p-6">
        <div className="flex items-center space-x-2 text-cyan-400 mb-4">
          <Sliders className="w-5 h-5" />
          <h3 className="font-bold text-white text-base">Model Hyperparameter Configuration</h3>
        </div>

        <form onSubmit={handleSaveConfig} className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
          <div>
            <label className="block text-xs font-semibold text-slate-300 mb-2">
              Delay Threshold (0.0 - 1.0)
            </label>
            <input
              type="number"
              step="0.05"
              min="0.1"
              max="0.9"
              value={delayThreshold}
              onChange={(e) => setDelayThreshold(e.target.value)}
              className="w-full px-3 py-2 bg-slate-900 border border-slate-700 rounded-xl text-sm text-slate-200 focus:outline-none focus:border-cyan-500"
            />
            <span className="text-[11px] text-slate-500 mt-1 block">Decision boundary for delay flag</span>
          </div>

          <div>
            <label className="block text-xs font-semibold text-slate-300 mb-2">
              Learning Rate
            </label>
            <input
              type="number"
              step="0.0001"
              value={learningRate}
              onChange={(e) => setLearningRate(e.target.value)}
              className="w-full px-3 py-2 bg-slate-900 border border-slate-700 rounded-xl text-sm text-slate-200 focus:outline-none focus:border-cyan-500"
            />
            <span className="text-[11px] text-slate-500 mt-1 block">Optimizer learning rate (Adam)</span>
          </div>

          <div>
            <label className="block text-xs font-semibold text-slate-300 mb-2">
              Epochs
            </label>
            <input
              type="number"
              value={epochs}
              onChange={(e) => setEpochs(e.target.value)}
              className="w-full px-3 py-2 bg-slate-900 border border-slate-700 rounded-xl text-sm text-slate-200 focus:outline-none focus:border-cyan-500"
            />
            <span className="text-[11px] text-slate-500 mt-1 block">Maximum training epochs</span>
          </div>

          <div>
            <label className="block text-xs font-semibold text-slate-300 mb-2">
              Batch Size
            </label>
            <input
              type="number"
              value={batchSize}
              onChange={(e) => setBatchSize(e.target.value)}
              className="w-full px-3 py-2 bg-slate-900 border border-slate-700 rounded-xl text-sm text-slate-200 focus:outline-none focus:border-cyan-500"
            />
            <span className="text-[11px] text-slate-500 mt-1 block">DataLoader batch size</span>
          </div>

          <div className="lg:col-span-4 flex justify-end">
            <button
              type="submit"
              disabled={savingConfig}
              className="px-6 py-2.5 bg-cyan-500 hover:bg-cyan-400 disabled:opacity-40 text-slate-950 font-bold rounded-xl text-xs flex items-center space-x-2 transition-colors shadow-lg shadow-cyan-500/20"
            >
              <Save className="w-4 h-4" />
              <span>{savingConfig ? "Saving..." : "Save Configuration"}</span>
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
