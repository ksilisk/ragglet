import "@fontsource/ibm-plex-sans/400.css";
import "@fontsource/ibm-plex-sans/500.css";
import "@fontsource/ibm-plex-sans/600.css";
import "@fontsource/ibm-plex-mono/400.css";
import "@fontsource/ibm-plex-mono/500.css";
import "./ui/base.css";

import * as Tooltip from "@radix-ui/react-tooltip";
import { QueryClientProvider } from "@tanstack/react-query";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { RouterProvider } from "react-router";
import { createQueryClient } from "./api/queries";
import { I18nProvider } from "./i18n/i18n";
import { router } from "./routes";
import { ConfirmProvider } from "./ui/Dialog";

const queryClient = createQueryClient();

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <I18nProvider>
      <QueryClientProvider client={queryClient}>
        <Tooltip.Provider delayDuration={300}>
          <ConfirmProvider>
            <RouterProvider router={router} />
          </ConfirmProvider>
        </Tooltip.Provider>
      </QueryClientProvider>
    </I18nProvider>
  </StrictMode>,
);
