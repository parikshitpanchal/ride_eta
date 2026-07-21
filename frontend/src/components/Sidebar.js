"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useAuth } from "@/context/AuthContext";
import {
  Table,
  BarChart3,
  Activity,
  Users,
  Sliders,
  Car,
  LogOut,
  LogIn,
  Shield,
  User,
} from "lucide-react";

export default function Sidebar() {
  const pathname = usePathname();
  const router = useRouter();
  const { user, role, logout, isAuthenticated } = useAuth();

  const navItems = [
    { label: "Predictions", href: "/predictions", icon: Table, roles: ["admin", "data_scientist", "guest"] },
    { label: "Model Dashboard", href: "/dashboard", icon: BarChart3, roles: ["admin", "data_scientist", "guest"] },
    { label: "Training Monitor", href: "/training", icon: Activity, roles: ["admin", "data_scientist", "guest"] },
    { label: "Driver Analytics", href: "/drivers", icon: Users, roles: ["admin", "data_scientist", "guest"] },
    { label: "Admin Panel", href: "/admin", icon: Sliders, roles: ["admin"] },
  ];

  const filteredNavItems = navItems.filter((item) => {
    if (!isAuthenticated) return true; // Show all to guest/unauthenticated for preview
    if (item.roles.includes("admin") && role === "admin") return true;
    return item.roles.includes(role) ;
  });

  return (
    <aside className="w-64 bg-[#0d1322] border-r border-slate-800/80 flex flex-col justify-between h-screen sticky top-0 z-30">
      <div>
        {/* Logo Header */}
        <div className="p-6 border-b border-slate-800/80 flex items-center space-x-3">
          <div className="p-2.5 bg-cyan-500/10 border border-cyan-500/30 rounded-xl text-cyan-400">
            <Car className="w-6 h-6" />
          </div>
          <div>
            <h1 className="font-bold text-lg text-white tracking-wide">Ride ETA</h1>
            <p className="text-xs text-cyan-400 font-medium">ML Intelligence Engine</p>
          </div>
        </div>

        {/* User Profile Card */}
        <div className="p-4 m-3 bg-slate-900/80 border border-slate-800 rounded-xl">
          {isAuthenticated ? (
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2.5">
                <div className="w-8 h-8 rounded-full bg-cyan-500/20 border border-cyan-500/30 flex items-center justify-center text-cyan-400 font-bold text-xs">
                  {user?.username?.[0]?.toUpperCase() || "U"}
                </div>
                <div>
                  <p className="text-xs font-bold text-slate-100">{user?.username}</p>
                  <span className={`inline-block px-1.5 py-0.2 rounded text-[10px] font-semibold uppercase ${
                    role === "admin"
                      ? "bg-rose-500/20 text-rose-300 border border-rose-500/30"
                      : role === "data_scientist"
                      ? "bg-purple-500/20 text-purple-300 border border-purple-500/30"
                      : "bg-emerald-500/20 text-emerald-300 border border-emerald-500/30"
                  }`}>
                    {role === "data_scientist" ? "Data Sci" : role}
                  </span>
                </div>
              </div>
              <button
                onClick={logout}
                title="Sign Out"
                className="p-1.5 text-slate-400 hover:text-rose-400 transition-colors"
              >
                <LogOut className="w-4 h-4" />
              </button>
            </div>
          ) : (
            <Link
              href="/login"
              className="flex items-center justify-between w-full py-1 text-xs font-semibold text-cyan-400 hover:text-cyan-300"
            >
              <div className="flex items-center space-x-2">
                <LogIn className="w-4 h-4" />
                <span>Sign In / Select Role</span>
              </div>
            </Link>
          )}
        </div>

        {/* Nav Links */}
        <nav className="px-3 py-2 space-y-1.5">
          {filteredNavItems.map((item) => {
            const Icon = item.icon;
            const isActive = pathname === item.href || (item.href !== "/predictions" && pathname.startsWith(item.href));

            return (
              <Link
                key={item.href}
                href={item.href}
                className={`flex items-center space-x-3 px-4 py-3 rounded-xl text-sm font-medium transition-all duration-150 ${
                  isActive
                    ? "bg-cyan-500/15 text-cyan-300 border border-cyan-500/30 shadow-lg shadow-cyan-500/5"
                    : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/40"
                }`}
              >
                <Icon className={`w-5 h-5 ${isActive ? "text-cyan-400" : "text-slate-400"}`} />
                <span>{item.label}</span>
              </Link>
            );
          })}
        </nav>
      </div>

      {/* Footer System Status */}
      <div className="p-4 m-3 bg-slate-900/60 border border-slate-800 rounded-xl text-[11px] text-slate-500">
        <div className="flex items-center space-x-2 font-semibold text-emerald-400">
          <Shield className="w-3.5 h-3.5 text-emerald-400" />
          <span>RBAC Security Active</span>
        </div>
        <p className="mt-1 text-[10px]">PyTorch & PostgreSQL Integrated</p>
      </div>
    </aside>
  );
}
