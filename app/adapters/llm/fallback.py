# app/adapters/llm/fallback_adapter.py
from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, List, Optional, Tuple

from app.domain import errors as de
from app.domain.concession_policy import DebateState
from app.domain.models import Conversation, Message
from app.domain.ports.llm import LLMPort


class FallbackLLM(LLMPort):
    def __init__(
        self,
        primary: LLMPort,
        secondary: LLMPort,
        *,
        per_provider_timeout_s: float = 15.0,
        mode: str = 'sequential',  # "sequential" | "hedged"
        logger: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.primary = primary
        self.secondary = secondary
        self.timeout = per_provider_timeout_s
        self.mode = mode
        self.log = logger or (lambda _msg: None)

    # ---- Public API expected by your service ----
    async def generate(self, conversation: Conversation, state: DebateState) -> str:
        return await self._invoke(lambda p: p.generate(conversation, state))

    async def debate(self, messages: List[Message], state: DebateState) -> str:
        return await self._invoke(lambda p: p.debate(messages, state))

    # ---- Internals ----
    async def _invoke(self, fn_builder: Callable[[LLMPort], Awaitable[str]]) -> str:
        if self.mode == 'hedged':
            return await self._hedged_call(fn_builder)
        return await self._sequential_call(fn_builder)

    async def _try_provider(
        self,
        label: str,
        provider: LLMPort,
        fn_builder: Callable[[LLMPort], Awaitable[str]],
    ) -> Tuple[bool, str | Exception]:
        """
        Returns (ok, result_or_exc). Does NOT raise.
        Maps low-level timeouts/errors into domain errors.
        """
        try:
            self.log(f'LLM {label}: start')
            result = await asyncio.wait_for(fn_builder(provider), timeout=self.timeout)
            return True, result
        except asyncio.TimeoutError:
            err = de.LLMTimeout(f'{label} provider timed out after {self.timeout:.2f}s')
            self.log(f'LLM {label}: timeout -> {err}')
            return False, err
        except Exception as e:
            err = de.LLMServiceError(
                f'{label} provider failed: {type(e).__name__}: {e}'
            )
            self.log(f'LLM {label}: failure -> {err}')
            return False, err

    def _raise_combined(self, errs: List[Exception]) -> None:
        # Prefer service errors if any; otherwise surface timeout
        non_timeouts = [e for e in errs if not isinstance(e, de.LLMTimeout)]
        if non_timeouts:
            detail = '; '.join(str(e) for e in non_timeouts)
            raise de.LLMServiceError(f'Both LLM providers failed. Details: {detail}')
        detail = '; '.join(str(e) for e in errs)
        raise de.LLMTimeout(f'Both LLM providers timed out. Details: {detail}')

    async def _sequential_call(
        self, fn_builder: Callable[[LLMPort], Awaitable[str]]
    ) -> str:
        ok1, r1 = await self._try_provider('primary', self.primary, fn_builder)
        if ok1:
            return r1  # type: ignore[return-value]

        ok2, r2 = await self._try_provider('secondary', self.secondary, fn_builder)
        if ok2:
            return r2  # type: ignore[return-value]

        # Both failed
        self._raise_combined([r1, r2])  # type: ignore[arg-type]
        raise AssertionError('unreachable')  # for type checkers

    async def _hedged_call(
        self, fn_builder: Callable[[LLMPort], Awaitable[str]]
    ) -> str:
        """
        Start primary immediately and secondary after a small delay.
        Return the first success; if both fail, raise a domain error.
        """
        loop = asyncio.get_event_loop()

        async def primary_coro():
            return await self._try_provider('primary', self.primary, fn_builder)

        async def secondary_coro():
            await asyncio.sleep(self.hedge_delay_s)
            return await self._try_provider('secondary', self.secondary, fn_builder)

        tasks = {loop.create_task(primary_coro()), loop.create_task(secondary_coro())}
        errors: List[Exception] = []

        while tasks:
            done, pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED
            )
            for t in done:
                ok, res = await t  # res: str | Exception
                if ok:
                    # Cancel the other task(s) and return
                    for p in pending:
                        p.cancel()
                    return res  # type: ignore[return-value]
                else:
                    errors.append(res)  # type: ignore[arg-type]
            tasks = pending

        # If we exit the loop, both failed
        self._raise_combined(errors)
        raise AssertionError('unreachable')
