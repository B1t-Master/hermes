import { createContext, useCallback, useContext, useMemo, useState } from "react";
import type { ReactNode } from "react";
import { api } from "./api";
import type { Role } from "./types";

interface AuthValue {
  token: string | null;
  role: Role | null;
  login: (email: string, password: string) => Promise<void>;
  register: (data: {
    first_name: string;
    last_name?: string;
    email: string;
    password: string;
  }) => Promise<void>;
  anonymous: (firstName: string) => Promise<void>;
  agentLogin: (username: string, password: string) => Promise<void>;
  logout: () => void;
}

const AuthContext = createContext<AuthValue | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [token, setToken] = useState<string | null>(() =>
    localStorage.getItem("hermes_token"),
  );
  const [role, setRole] = useState<Role | null>(() => {
    const stored = localStorage.getItem("hermes_role");
    return stored === "passenger" || stored === "agent" ? stored : null;
  });

  const persist = useCallback((t: string, r: Role) => {
    localStorage.setItem("hermes_token", t);
    localStorage.setItem("hermes_role", r);
    setToken(t);
    setRole(r);
  }, []);

  const login = useCallback(
    async (email: string, password: string) => {
      const res = await api.login({ email, password });
      persist(res.access_token, res.role);
    },
    [persist],
  );

  const register = useCallback(
    async (data: {
      first_name: string;
      last_name?: string;
      email: string;
      password: string;
    }) => {
      await api.register(data);
      await login(data.email, data.password);
    },
    [login],
  );

  const anonymous = useCallback(
    async (firstName: string) => {
      const res = await api.anonymous({ first_name: firstName });
      persist(res.access_token, res.role);
    },
    [persist],
  );

  const agentLogin = useCallback(
    async (username: string, password: string) => {
      const res = await api.agentLogin({ username, password });
      persist(res.access_token, res.role);
    },
    [persist],
  );

  const logout = useCallback(() => {
    localStorage.removeItem("hermes_token");
    localStorage.removeItem("hermes_role");
    setToken(null);
    setRole(null);
  }, []);

  const value = useMemo<AuthValue>(
    () => ({ token, role, login, register, anonymous, agentLogin, logout }),
    [token, role, login, register, anonymous, agentLogin, logout],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthValue {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used inside <AuthProvider>");
  return ctx;
}
