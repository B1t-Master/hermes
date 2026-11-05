import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from "react";
import type { ReactNode } from "react";
import { api } from "./api";
import type { Role } from "./types";

/** Renew this long before the token expires. */
const REFRESH_LEAD_MS = 5 * 60 * 1000;

/** Read the JWT `exp` claim (milliseconds), or null when unreadable. */
function tokenExpiry(token: string): number | null {
  try {
    const payload = token.split(".")[1];
    if (!payload) return null;
    const base64 = payload.replace(/-/g, "+").replace(/_/g, "/");
    const padded = base64.padEnd(Math.ceil(base64.length / 4) * 4, "=");
    const claims = JSON.parse(atob(padded)) as { exp?: unknown };
    return typeof claims.exp === "number" ? claims.exp * 1000 : null;
  } catch {
    return null;
  }
}

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

  /* Silently renew shortly before the token expires. A failed refresh means
   * the session can no longer be extended, so the user is logged out. */
  useEffect(() => {
    if (!token) return;
    const expiresAt = tokenExpiry(token);
    if (expiresAt === null) return;

    let cancelled = false;
    const timer = window.setTimeout(
      async () => {
        try {
          const res = await api.refresh(token);
          if (!cancelled) persist(res.access_token, res.role);
        } catch {
          if (!cancelled) logout();
        }
      },
      Math.max(0, expiresAt - Date.now() - REFRESH_LEAD_MS),
    );

    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [token, persist, logout]);

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
