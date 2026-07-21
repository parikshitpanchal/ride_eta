"use client";

import { useState, useEffect, useCallback } from "react";
import { getDrivers, getTopDrivers, getWorstDrivers } from "@/lib/api";
import {
  Users,
  Award,
  AlertOctagon,
  Search,
  Star,
  ShieldCheck,
  Zap,
  ChevronLeft,
  ChevronRight,
  ArrowUpDown,
} from "lucide-react";

export default function DriversPage() {
  const [drivers, setDrivers] = useState([]);
  const [topDrivers, setTopDrivers] = useState([]);
  const [worstDrivers, setWorstDrivers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Pagination & Search
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(25);
  const [search, setSearch] = useState("");
  const [sortBy, setSortBy] = useState("delay_rate");
  const [sortOrder, setSortOrder] = useState("desc");
  const [totalPages, setTotalPages] = useState(1);
  const [totalCount, setTotalCount] = useState(0);

  useEffect(() => {
    let isMounted = true;
    async function loadData() {
      setLoading(true);
      setError(null);
      try {
        const [driverData, topData, worstData] = await Promise.all([
          getDrivers({ page, pageSize, search, sortBy, sortOrder }),
          getTopDrivers(),
          getWorstDrivers(),
        ]);

        if (isMounted) {
          setDrivers(driverData.items || []);
          setTotalPages(driverData.total_pages || 1);
          setTotalCount(driverData.total || 0);
          setTopDrivers(topData || []);
          setWorstDrivers(worstData || []);
        }
      } catch (err) {
        if (isMounted) setError(err.message || "Failed to load driver data");
      } finally {
        if (isMounted) setLoading(false);
      }
    }

    loadData();
    return () => {
      isMounted = false;
    };
  }, [page, pageSize, search, sortBy, sortOrder]);

  const toggleSort = (field) => {
    if (sortBy === field) {
      setSortOrder(sortOrder === "asc" ? "desc" : "asc");
    } else {
      setSortBy(field);
      setSortOrder("desc");
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-white tracking-tight">Driver Performance Analytics</h1>
        <p className="text-sm text-slate-400 mt-1">
          Historical driver delay rates, rating metrics, and PyTorch feature engineering scores
        </p>
      </div>

      {error && (
        <div className="p-4 bg-rose-500/10 border border-rose-500/30 rounded-xl text-rose-300 text-sm">
          {error}
        </div>
      )}

      {/* Highlights: Top & Worst Performers */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Top Reliable Drivers */}
        <div className="glass-card p-5 border-t-4 border-t-emerald-500">
          <h2 className="text-sm font-bold text-white mb-3 flex items-center space-x-2">
            <Award className="w-4 h-4 text-emerald-400" />
            <span>Top Reliable Drivers (Lowest Delay Rate)</span>
          </h2>
          <div className="space-y-2">
            {topDrivers.slice(0, 5).map((d) => (
              <div
                key={d.id}
                className="flex items-center justify-between p-2.5 bg-slate-900/60 border border-slate-800 rounded-xl text-xs"
              >
                <div className="flex items-center space-x-3">
                  <span className="font-mono font-bold text-slate-200">{d.driver_id}</span>
                  <div className="flex items-center space-x-1 text-amber-400">
                    <Star className="w-3 h-3 fill-amber-400" />
                    <span>{d.avg_rating?.toFixed(1)}</span>
                  </div>
                </div>
                <div className="flex items-center space-x-4">
                  <span className="text-slate-400">{d.total_rides} rides</span>
                  <span className="px-2 py-0.5 bg-emerald-500/20 text-emerald-300 font-mono font-bold rounded">
                    {(d.delay_rate * 100).toFixed(1)}% delay
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Highest Delay Rate Drivers */}
        <div className="glass-card p-5 border-t-4 border-t-rose-500">
          <h2 className="text-sm font-bold text-white mb-3 flex items-center space-x-2">
            <AlertOctagon className="w-4 h-4 text-rose-400" />
            <span>Highest Delay Rate Drivers (High Risk)</span>
          </h2>
          <div className="space-y-2">
            {worstDrivers.slice(0, 5).map((d) => (
              <div
                key={d.id}
                className="flex items-center justify-between p-2.5 bg-slate-900/60 border border-slate-800 rounded-xl text-xs"
              >
                <div className="flex items-center space-x-3">
                  <span className="font-mono font-bold text-slate-200">{d.driver_id}</span>
                  <div className="flex items-center space-x-1 text-amber-400">
                    <Star className="w-3 h-3 fill-amber-400" />
                    <span>{d.avg_rating?.toFixed(1)}</span>
                  </div>
                </div>
                <div className="flex items-center space-x-4">
                  <span className="text-slate-400">{d.total_rides} rides</span>
                  <span className="px-2 py-0.5 bg-rose-500/20 text-rose-300 font-mono font-bold rounded">
                    {(d.delay_rate * 100).toFixed(1)}% delay
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Filter & Search Bar */}
      <div className="glass-card p-4 flex flex-col md:flex-row items-center justify-between gap-4">
        <div className="relative w-full md:w-80">
          <Search className="absolute left-3 top-2.5 w-4 h-4 text-slate-400" />
          <input
            type="text"
            placeholder="Search Driver ID (e.g. DRV_0001)..."
            value={search}
            onChange={(e) => {
              setSearch(e.target.value);
              setPage(1);
            }}
            className="w-full pl-9 pr-4 py-2 bg-slate-900/80 border border-slate-700/60 rounded-xl text-sm text-slate-200 placeholder-slate-500 focus:outline-none focus:border-cyan-500"
          />
        </div>

        <div className="flex items-center space-x-3 w-full md:w-auto">
          <select
            value={pageSize}
            onChange={(e) => {
              setPageSize(Number(e.target.value));
              setPage(1);
            }}
            className="bg-slate-900/80 border border-slate-700/60 rounded-xl px-3 py-2 text-xs text-slate-300 focus:outline-none focus:border-cyan-500"
          >
            <option value={15}>15 per page</option>
            <option value={25}>25 per page</option>
            <option value={50}>50 per page</option>
            <option value={100}>100 per page</option>
          </select>
        </div>
      </div>

      {/* Drivers Table */}
      <div className="glass-card overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm text-slate-300">
            <thead className="bg-slate-900/90 text-xs uppercase text-slate-400 font-semibold border-b border-slate-800">
              <tr>
                <th
                  onClick={() => toggleSort("driver_id")}
                  className="px-4 py-3.5 cursor-pointer hover:text-slate-200"
                >
                  Driver ID
                </th>
                <th
                  onClick={() => toggleSort("total_rides")}
                  className="px-4 py-3.5 cursor-pointer hover:text-slate-200"
                >
                  Total Rides
                </th>
                <th className="px-4 py-3.5">Delayed Rides</th>
                <th
                  onClick={() => toggleSort("delay_rate")}
                  className="px-4 py-3.5 cursor-pointer hover:text-slate-200 text-cyan-400"
                >
                  Delay Rate
                </th>
                <th
                  onClick={() => toggleSort("avg_delay_minutes")}
                  className="px-4 py-3.5 cursor-pointer hover:text-slate-200"
                >
                  Avg Delay
                </th>
                <th
                  onClick={() => toggleSort("avg_rating")}
                  className="px-4 py-3.5 cursor-pointer hover:text-slate-200"
                >
                  Rating
                </th>
                <th className="px-4 py-3.5 text-center">Reliability</th>
                <th className="px-4 py-3.5 text-center">Risk Score</th>
                <th className="px-4 py-3.5 text-center">Efficiency</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 font-mono text-xs">
              {loading ? (
                Array.from({ length: 8 }).map((_, idx) => (
                  <tr key={idx} className="animate-pulse">
                    <td className="px-4 py-4"><div className="h-4 bg-slate-800 rounded w-20"></div></td>
                    <td className="px-4 py-4"><div className="h-4 bg-slate-800 rounded w-12"></div></td>
                    <td className="px-4 py-4"><div className="h-4 bg-slate-800 rounded w-12"></div></td>
                    <td className="px-4 py-4"><div className="h-4 bg-slate-800 rounded w-16"></div></td>
                    <td className="px-4 py-4"><div className="h-4 bg-slate-800 rounded w-16"></div></td>
                    <td className="px-4 py-4"><div className="h-4 bg-slate-800 rounded w-12"></div></td>
                    <td className="px-4 py-4 text-center"><div className="h-4 bg-slate-800 rounded w-12 mx-auto"></div></td>
                    <td className="px-4 py-4 text-center"><div className="h-4 bg-slate-800 rounded w-12 mx-auto"></div></td>
                    <td className="px-4 py-4 text-center"><div className="h-4 bg-slate-800 rounded w-12 mx-auto"></div></td>
                  </tr>
                ))
              ) : drivers.length === 0 ? (
                <tr>
                  <td colSpan={9} className="text-center py-12 text-slate-500 font-sans">
                    No drivers found.
                  </td>
                </tr>
              ) : (
                drivers.map((d) => {
                  const delayPct = (d.delay_rate * 100).toFixed(1);
                  return (
                    <tr key={d.id} className="hover:bg-slate-800/40 transition-colors">
                      <td className="px-4 py-3.5 font-bold text-slate-200">{d.driver_id}</td>
                      <td className="px-4 py-3.5 font-sans">{d.total_rides}</td>
                      <td className="px-4 py-3.5 font-sans">{d.delayed_rides}</td>
                      <td className="px-4 py-3.5">
                        <span
                          className={`px-2 py-0.5 rounded font-bold ${
                            d.delay_rate > 0.3
                              ? "bg-rose-500/20 text-rose-300"
                              : d.delay_rate > 0.15
                              ? "bg-amber-500/20 text-amber-300"
                              : "bg-emerald-500/20 text-emerald-300"
                          }`}
                        >
                          {delayPct}%
                        </span>
                      </td>
                      <td className="px-4 py-3.5">{d.avg_delay_minutes?.toFixed(1)} min</td>
                      <td className="px-4 py-3.5 text-amber-400 font-sans font-semibold">
                        ★ {d.avg_rating?.toFixed(1)}
                      </td>
                      <td className="px-4 py-3.5 text-center text-cyan-400">
                        {d.reliability_score?.toFixed(2)}
                      </td>
                      <td className="px-4 py-3.5 text-center text-purple-400">
                        {d.risk_score?.toFixed(2)}
                      </td>
                      <td className="px-4 py-3.5 text-center text-emerald-400">
                        {d.efficiency_score?.toFixed(2)}
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>

        {/* Pagination Footer */}
        <div className="p-4 bg-slate-900/60 border-t border-slate-800/80 flex flex-col sm:flex-row items-center justify-between gap-4 text-xs text-slate-400 font-sans">
          <div>
            Showing <span className="font-semibold text-slate-200">{drivers.length}</span> of{" "}
            <span className="font-semibold text-slate-200">{totalCount.toLocaleString()}</span> drivers
          </div>
          <div className="flex items-center space-x-2">
            <button
              onClick={() => setPage((p) => Math.max(1, p - 1))}
              disabled={page === 1 || loading}
              className="p-2 bg-slate-800 hover:bg-slate-700 disabled:opacity-40 disabled:cursor-not-allowed rounded-lg transition-colors"
            >
              <ChevronLeft className="w-4 h-4" />
            </button>
            <span className="px-3 py-1 bg-slate-800/60 border border-slate-700/60 rounded-lg text-slate-200 font-mono">
              Page {page} of {totalPages}
            </span>
            <button
              onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
              disabled={page >= totalPages || loading}
              className="p-2 bg-slate-800 hover:bg-slate-700 disabled:opacity-40 disabled:cursor-not-allowed rounded-lg transition-colors"
            >
              <ChevronRight className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
